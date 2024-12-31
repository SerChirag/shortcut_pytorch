# Use a lightweight base image with Conda installed
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    wget \
    python3-pip \
    python3-dev 

# Create a working directory
RUN pip install --upgrade pip
RUN pip install accelerate==1.2.0 \
    && pip install diffusers==0.31.0 \
    && pip install torch==2.5.1 \
    && pip install timm==1.0.12 \
    && pip install torchmetrics[image] \
    && pip install matplotlib \
    && pip install pandas==2.2.3 \
    && pip install fastparquet==2024.11.0 \
    && pip install pytorch-lightning \
    && pip install tensorboard \
    && pip install wandb

# 1. visualize logs(on server):
# tensorboard --logdir=tb_logs/shortcut_model --port 6006
# 2. on own pc:
# ssh -N -f -L localhost:16006:localhost:6006 r.khafizov@10.16.88.93 
# if port is already in use, do:
    # 1. sudo netstat -tulpn | grep :16006
    # 2. kill observed process id `kill "id"`