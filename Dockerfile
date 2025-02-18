# Use PyTorch 2.1 GPU base image with Python 3.8 and CUDA 11.8 on Ubuntu 22.04
FROM pytorch/pytorch:2.1-gpu-py38-cu118-ubuntu22.04

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/FunAudioLLM/InspireMusic"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/pytorch:2.1-gpu-py38-cu118-ubuntu22.04"

# Set the working directory
WORKDIR /workspace/InspireMusic
# Copy the current directory contents into the container at /workspace/Open-Sora
COPY . .

# inatall library dependencies
RUN apt-get update && apt-get install -y ffmpeg sox libsox-dev
RUN pip install --no-cache-dir -r requirements.txt

# install flash attention
RUN pip install flash-attn==2.6.3 --no-build-isolation

# download models
RUN mkdir -p /workspace/InspireMusic/pretrained_models
RUN cd /workspace/InspireMusic/pretrained_models
RUN git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-Long.git
RUN git clone https://modelscope.cn/models/iic/InspireMusic.git
RUN git clone https://modelscope.cn/models/iic/InspireMusic-1.5B.git
RUN git clone https://modelscope.cn/models/iic/InspireMusic-Base-24kHz.git
RUN git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz.git