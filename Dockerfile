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
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    sox \
    libsox-dev \
    git \
    git-lfs && \
    # Verificar que git esté instalado
    git --version && \
    # Limpiar después de instalar todo
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy the current directory contents into the container at /workspace/InspireMusic
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Install Google Cloud Storage library (needed for GCS upload)
RUN pip install google-cloud-storage

# Download models from HuggingFace (recommended, as they are public)
RUN mkdir -p /workspace/InspireMusic/pretrained_models && \
    cd /workspace/InspireMusic/pretrained_models && \
    git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git && \
    git clone https://huggingface.co/FunAudioLLM/InspireMusic.git && \
    git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B.git && \
    git clone https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz.git && \
    git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz.git

# Copy GCP credentials (ensure the file exists in the build context)
COPY gcp-credentials.json /workspace/InspireMusic/gcp-credentials.json

# Set environment variable for GCP credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/workspace/InspireMusic/gcp-credentials.json

# Set the entrypoint (optional, we'll override it in RunPod Serverless)
CMD ["bash"]