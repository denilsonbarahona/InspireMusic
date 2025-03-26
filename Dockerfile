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
    apt-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Establece Python 3.8 como predeterminado
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Actualiza pip
RUN python3 -m pip install --upgrade pip

# Instala PyTorch 2.0.1 con CUDA 11.8
RUN python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Instala huggingface_hub, transformers y safetensors
RUN python3 -m pip install huggingface_hub transformers safetensors

# Establece el directorio de trabajo
WORKDIR /workspace/InspireMusic

# Copia los archivos necesarios al contenedor, incluyendo README.md
COPY . .

# Instala el paquete inspiremusic y sus dependencias usando setup.py
RUN python3 -m pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu118

# install flash attention
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Crea el directorio para los modelos preentrenados
RUN mkdir -p /workspace/InspireMusic/pretrained_models

# Descarga los modelos usando huggingface_hub
RUN python3 -c "from huggingface_hub import snapshot_download; \
    print('Descargando InspireMusic-1.5B-Long...'); \
    snapshot_download(repo_id='FunAudioLLM/InspireMusic-1.5B-Long', local_dir='/workspace/InspireMusic/pretrained_models/InspireMusic-1.5B-Long');"

    
# Aplica el ajuste al archivo inspiremusic.yaml si existe
RUN if [ -f "/workspace/InspireMusic/pretrained_models/InspireMusic-1.5B-Long/inspiremusic.yaml" ]; then \
        sed -i -e "s/\.\.\/\.\.\///g" /workspace/InspireMusic/pretrained_models/InspireMusic-1.5B-Long/inspiremusic.yaml; \
    else \
        echo "Warning: inspiremusic.yaml not found in InspireMusic-1.5B-Long"; \
        exit 1; \
    fi

# Instala Matcha-TTS como un paquete
RUN cd /workspace/InspireMusic/third_party/Matcha-TTS && \
    python3 -m pip install .
    
# Comando por defecto para pruebas
CMD ["python3", "handler.py"]