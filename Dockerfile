# SAM3 + MatAnyone RunPod Serverless worker
# Built by .github/workflows/build.yml on GitHub Actions (amd64)

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
      python3.11 python3.11-venv python3-pip python3.11-dev \
      git curl ca-certificates \
      ffmpeg libsm6 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

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

CMD ["python", "-u", "handler.py"]
