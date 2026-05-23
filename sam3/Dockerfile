# SAM3 + MatAnyone RunPod Serverless worker
# Build: docker build -t bgless-sam3:dev -f runpod/sam3/Dockerfile .
# Test : docker run --gpus all -e HF_TOKEN=... -e R2_* ... -p 8000:8000 bgless-sam3:dev

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TORCH_HOME=/runpod-volume/torch-cache \
    SAM3_WEIGHTS_DIR=/runpod-volume/sam3 \
    MATANYONE_WEIGHTS_DIR=/runpod-volume/matanyone

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      git curl ca-certificates \
      ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace

# pip deps (cached layer)
COPY runpod/sam3/requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

# SAM3 + MatAnyone from git (gated weights NOT baked in; mount via network volume)
RUN pip install \
      "git+https://github.com/facebookresearch/sam3.git@main" \
      "git+https://github.com/pq-yang/MatAnyone.git@main"

# Handler source
COPY runpod/sam3/handler.py    /workspace/handler.py
COPY runpod/sam3/pipeline.py   /workspace/pipeline.py
COPY runpod/sam3/schema.py     /workspace/schema.py
COPY runpod/sam3/storage.py    /workspace/storage.py
COPY runpod/sam3/encoder.py    /workspace/encoder.py

# Healthcheck endpoint (RunPod calls /ping)
EXPOSE 8000

CMD ["python", "-u", "handler.py"]
