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
    && pip install tensorboard

# visualize logs(on server):
tensorboard --logdir=lightning_logs/ --port 6006
# on own pc:
ssh -N -f -L localhost:16006:localhost:6006 r.khafizov@10.16.90.17
