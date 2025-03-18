# Use PyTorch 2.1 GPU base image with Python 3.8 and CUDA 11.8 on Ubuntu 22.04
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Metainformation
LABEL org.opencontainers.image.source="https://github.com/FunAudioLLM/InspireMusic"
LABEL org.opencontainers.image.licenses="Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/pytorch:2.1.0-cuda11.8-cudnn8-runtime"

# Set the working directory
WORKDIR /workspace/InspireMusic

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsox-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy the current directory contents into the container at /workspace/InspireMusic
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Download models
RUN mkdir -p /workspace/InspireMusic/pretrained_models && \
    cd /workspace/InspireMusic/pretrained_models && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-Long.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-Base-24kHz.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz.git

# Set the entrypoint (optional, we'll override it in RunPod Serverless)
CMD ["bash"]