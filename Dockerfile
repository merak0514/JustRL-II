# JustRL2 training image.
#
# Built on the community Miles image, which already contains everything the training
# stack needs: PyTorch + CUDA, TransformerEngine, apex, Megatron-LM
# (radixark/Megatron-LM @ miles-main, checked out at /root/Megatron-LM), SGLang
# (sglang-miles branch at /sgl-workspace/sglang), Megatron-Bridge (used by the
# HF -> torch_dist converter) and Ray. See https://github.com/radixark/miles/tree/main/docker
# for how that image is produced and which tags exist.
#
#   docker build -t justrl2 .
#   docker run --gpus all --ipc=host --network=host -it \
#       -v /path/to/models:/workspace/JustRL2/models \
#       -v /path/to/datasets:/workspace/JustRL2/datasets \
#       -v /path/to/runs:/workspace/JustRL2/runs justrl2
#
# DSpark speculative decoding is NOT available in this image (it needs an SGLang build
# that carries the DSpark scheduler); leave DSPARK_DRAFT_MODEL_PATH empty.

ARG MILES_IMAGE=radixark/miles:dev
FROM ${MILES_IMAGE}

WORKDIR /workspace/JustRL2
COPY . .

# train.sh expects the two frameworks next to the repo root.
RUN ln -sfn /root/Megatron-LM Megatron-LM && \
    ln -sfn /sgl-workspace/sglang sglang

# The image already satisfies requirements.txt; install only what the recipe adds.
# hf_transfer backs the HF_HUB_ENABLE_HF_TRANSFER=1 below: huggingface_hub raises rather
# than falling back when the flag is set without the package installed.
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir "math-verify==0.9.0" "antlr4-python3-runtime" "hf_transfer"

# Dependencies are baked in; setup.sh only applies the Megatron patch (idempotent).
ENV SKIP_PIP_INSTALL=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["/bin/bash"]
