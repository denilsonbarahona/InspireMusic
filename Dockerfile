# Usa una imagen base oficial de NVIDIA con CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Establece variables de entorno
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias básicas del sistema, incluyendo Python y git
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3.8 \
    python3-pip \
    python3-dev \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Establece Python 3.8 como predeterminado
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Actualiza pip
RUN python3 -m pip install --upgrade pip

# Instala PyTorch 2.0.1 con CUDA 11.8
RUN python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Establece el directorio de trabajo
WORKDIR /workspace/InspireMusic

# Copia los archivos necesarios al contenedor, incluyendo README.md
COPY . .

# Instala el paquete inspiremusic y sus dependencias usando setup.py
RUN python3 -m pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Crea el directorio y descarga los modelos preentrenados
RUN mkdir -p /workspace/InspireMusic/pretrained_models
RUN cd /workspace/InspireMusic/pretrained_models
RUN git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git
RUN sed -i -e "s/\.\.\/\.\.\///g" /workspace/InspireMusic/pretrained_models/InspireMusic-1.5B-Long/inspiremusic.yaml

# Comando por defecto para pruebas
CMD ["python3", "handler.py"]