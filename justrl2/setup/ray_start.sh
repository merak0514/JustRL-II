MASTER_IP=$(getent ahostsv4 $MASTER_ADDR | head -1 | awk '{print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(hostname -I | awk '{print $1}')
fi
RAY_PORT=$(($MASTER_PORT + 1))
export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"
ray stop --force
pkill -9 ray || true
pkill -9 redis || true

RAY_TMPDIR="/tmp/ray/rank_${RANK}"
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-4000000000}
RAY_SPILL_MIN_FREE_PCT=${RAY_SPILL_MIN_FREE_PCT:-10}

select_ray_spill_dir() {
    local candidate use_pct
    local max_use_pct=$((100 - RAY_SPILL_MIN_FREE_PCT))

    if [ -n "${RAY_SPILL_DIR:-}" ]; then
        mkdir -p "${RAY_SPILL_DIR}" || return 1
        use_pct=$(df -P "${RAY_SPILL_DIR}" 2>/dev/null | awk 'NR==2 {gsub("%", "", $5); print $5}')
        if [ -n "${use_pct}" ] && [ "${use_pct}" -le "${max_use_pct}" ]; then
            echo "${RAY_SPILL_DIR}"
            return 0
        fi
        echo "WARN: RAY_SPILL_DIR=${RAY_SPILL_DIR} usage=${use_pct:-unknown}% exceeds limit ${max_use_pct}%." >&2
    fi

    for candidate in \
        "${RAY_SPILL_DIR_ALT:-/tmp/ray_spill/rank_${RANK}}" \
        "${WORK_DIR}/ray_spill/rank_${RANK}" \
        "${HOME}/ray_spill/rank_${RANK}"; do
        mkdir -p "${candidate}" || continue
        use_pct=$(df -P "${candidate}" 2>/dev/null | awk 'NR==2 {gsub("%", "", $5); print $5}')
        if [ -n "${use_pct}" ] && [ "${use_pct}" -le "${max_use_pct}" ]; then
            echo "${candidate}"
            return 0
        fi
        echo "WARN: skip Ray spill dir ${candidate}, usage=${use_pct:-unknown}% exceeds limit ${max_use_pct}%." >&2
    done

    candidate="/tmp/ray_spill/rank_${RANK}"
    mkdir -p "${candidate}" || return 1
    echo "WARN: all Ray spill candidates are over threshold, fallback to ${candidate}." >&2
    echo "${candidate}"
}

RAY_SPILL_DIR=$(select_ray_spill_dir)
rm -rf /tmp/ray $RAY_TMPDIR
mkdir -p $RAY_TMPDIR
mkdir -p $RAY_SPILL_DIR

# Ephemeral-storage watchdog. Multi-day runs have been killed for exceeding the pod's
# local-disk quota, so every 30 min print the top local-disk consumers (to locate the
# source next time) and truncate oversized log files once past the soft limit.
# DISABLE_DISK_WATCHDOG=1 turns it off; EPHEMERAL_SOFT_LIMIT_MB defaults to 150 GB.
if [ "${DISABLE_DISK_WATCHDOG:-0}" != "1" ]; then
(
    while sleep 1800; do
        echo "[disk-watchdog rank${RANK}] $(date '+%F %T') local disk top:"
        du -xm -d 2 /tmp /data 2>/dev/null | sort -rn | head -12
        used_mb=$(du -xsm /tmp /data 2>/dev/null | awk '{s+=$1} END {print s+0}')
        if [ "${used_mb}" -gt "${EPHEMERAL_SOFT_LIMIT_MB:-153600}" ]; then
            echo "[disk-watchdog rank${RANK}] ${used_mb}MB > soft limit, truncating oversized log files"
            find /tmp/ray "${RAY_LOG_DIR:-/tmp}" -type f \( -name '*.log' -o -name '*.out' -o -name '*.err' \) -size +256M \
                -exec sh -c 'echo "[disk-watchdog] truncate: $1 ($(du -m "$1" | cut -f1)MB)"; truncate -s 0 "$1"' _ {} \; 2>/dev/null
        fi
    done
) &
fi

echo "ray starting at rank $RANK"
echo "RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY}"
echo "RAY_SPILL_DIR=${RAY_SPILL_DIR}"
if [ $RANK -eq 0 ]; then
    ray start --head \
    --port=$RAY_PORT \
    --temp-dir=$RAY_TMPDIR \
    --num-gpus=${GPUS_PER_NODE} \
    --num-cpus=80 \
    --object-store-memory=${RAY_OBJECT_STORE_MEMORY} \
    --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"'$RAY_SPILL_DIR'\"}}","max_io_workers":4,"object_spilling_threshold":0.8}' \
    --include-dashboard=true \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    if [ $? -ne 0 ]; then
        echo "ERROR: failed to start the head node!"
        exit 1
    fi
    echo "Head node is ready"
else
    sleep 10
    ray start --address="$RAY_ADDRESS" \
    --num-gpus=${GPUS_PER_NODE} \
    --node-ip-address=$(hostname -i) \
    --num-cpus=80 \
    --object-store-memory=${RAY_OBJECT_STORE_MEMORY} \
    --disable-usage-stats \
    --block
    echo "Worker node connected"
fi
if [ $RANK -eq 0 ]; then
    echo "Waiting for all ${WORLD_SIZE} nodes to join the Ray cluster..."
    MAX_WAIT=600
    WAITED=0
    INTERVAL=15
    while [ $WAITED -lt $MAX_WAIT ]; do
        NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || true)
        echo "Active nodes: ${NODE_COUNT}/${WORLD_SIZE} (waited ${WAITED}s)"
        if [ "$NODE_COUNT" -ge "$WORLD_SIZE" ]; then
            echo "All ${WORLD_SIZE} nodes are ready!"
            break
        fi
        sleep $INTERVAL
        WAITED=$((WAITED + INTERVAL))
    done
    if [ "$NODE_COUNT" -lt "$WORLD_SIZE" ]; then
        echo "WARNING: timed out! Only ${NODE_COUNT}/${WORLD_SIZE} nodes are active"
        ray status
    fi

    echo "Waiting for the Ray dashboard to start..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f http://127.0.0.1:8265/api/version > /dev/null 2>&1; then
            echo "Ray dashboard is ready"
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 2
    done
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "WARNING: the Ray dashboard may not be fully up, continuing anyway..."
    fi

    ray status
fi