# SAM3 + MatAnyone RunPod Serverless worker
# Python 3.10 (Ubuntu 22.04 default) — cchardet (sam3 transitive dep) lacks 3.11 wheels.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/hf-cache \
    TORCH_HOME=/workspace/torch-cache \
    SAM3_WEIGHTS_DIR=/workspace/sam3 \
    MATANYONE_WEIGHTS_DIR=/workspace/matanyone

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev \
      git curl ca-certificates \
      ffmpeg libsm6 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY sam3/requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

# SAM3 + MatAnyone from git (gated weights NOT baked in; mounted via network volume)
RUN pip install \
      "git+https://github.com/facebookresearch/sam3.git@main" \
      "git+https://github.com/pq-yang/MatAnyone.git@main"

COPY sam3/handler.py    /workspace/handler.py
COPY sam3/pipeline.py   /workspace/pipeline.py
COPY sam3/schema.py     /workspace/schema.py
COPY sam3/storage.py    /workspace/storage.py
COPY sam3/encoder.py    /workspace/encoder.py

EXPOSE 8000

CMD ["python3", "-u", "handler.py"]
