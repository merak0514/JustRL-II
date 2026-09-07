# Copy Ray session logs out of the (possible) ephemeral /tmp. Override RAY_LOG_DST for your
# persistent storage; no hardcoded cluster path.
RAY_LOG_DST=${RAY_LOG_DST:-/tmp/ray_snapshot}
mkdir -p "$RAY_LOG_DST"
cp -r /tmp/ray "$RAY_LOG_DST" 2>/dev/null || true
