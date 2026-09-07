# SKIP_PIP_INSTALL=1 : skip all pip installs (use when the base image already has deps)
# PIP_INDEX_URL / PIP_INDEX_URL_OVERRIDE : set a mirror appropriate for your network.
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  rm -f /etc/xdg/pip/pip.conf /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf 2>/dev/null || true
  rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED 2>/dev/null || true
  rm -f /etc/pip/constraint.txt 2>/dev/null || true
  touch /etc/pip/constraint.txt

  # Default to the public PyPI; override with PIP_INDEX_URL (e.g. a mirror) for your cluster.
  pip config set global.index-url "${PIP_INDEX_URL:-https://pypi.org/simple}"

  # sglang-kernel >= 0.3.20 is required by the sglang fork (see models/sglang/README.md).
  cur_sgl_ver=$(pip show sgl-kernel 2>/dev/null | awk '/^Version:/{print $2}')
  if [ -z "$cur_sgl_ver" ]; then
    cur_sgl_ver=$(pip show sglang-kernel 2>/dev/null | awk '/^Version:/{print $2}')
  fi
  if [ -z "$cur_sgl_ver" ] || python3 -c "from packaging.version import Version; exit(0 if Version('$cur_sgl_ver') < Version('0.3.20') else 1)" 2>/dev/null; then
    pip install -U "sgl-kernel>=0.3.20" || echo "[setup.sh] sgl-kernel install failed, continuing..."
  else
    echo "[setup.sh] sgl-kernel $cur_sgl_ver >= 0.3.20, skip upgrade."
  fi

  pip install timeout_decorator polars seaborn
  # math-verify is the public PyPI package used for math answer verification (#12 eval).
  pip install math-verify antlr4-python3-runtime -U
  pip install grpcio grpcio-tools protobuf loguru
  pip install swanlab
  echo "[setup.sh] pip installs completed."
else
  echo "[setup.sh] SKIP_PIP_INSTALL=1, skipping all pip installs."
fi

# Apply the miles patch set to the Megatron-LM submodule. The patch is what wires
# miles' scalar value-head critic and MiniCPM5 support into the Megatron rollout —
# it MUST be applied or the critic's output_layer/LM-head handling breaks. The
# marker file makes the patch idempotent (re-running setup.sh is a no-op).
cd Megatron-LM
OLD_MARKER=".miles_megatron_patch_v0.5.7_applied"
PATCH_MARKER=".miles_megatron_patch_latest_applied"
if [ -f "${OLD_MARKER}" ] && [ ! -f "${PATCH_MARKER}" ]; then
  mv "${OLD_MARKER}" "${PATCH_MARKER}"
  echo "migrated patch marker: ${OLD_MARKER} -> ${PATCH_MARKER}"
fi
if [ -f "${PATCH_MARKER}" ]; then
  echo "skip megatron.patch (marker exists: ${PATCH_MARKER})"
else
  patch -p1 -N --batch < "$WORK_DIR/docker/patch/latest/megatron.patch" || true
  touch "${PATCH_MARKER}"
  echo "applied megatron.patch (created marker: ${PATCH_MARKER})"
fi
