FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bison \
    build-essential \
    ca-certificates \
    curl \
    flex \
    libfl-dev \
    git \
    libcairo2-dev \
    libboost-all-dev \
    libz-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    zlib1g-dev \
    wget \
    matplotlib \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install cmake==3.29.6 \
 && python3 -m pip install 'numpy<2' \
 && python3 -m pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace

CMD ["/bin/bash"]
