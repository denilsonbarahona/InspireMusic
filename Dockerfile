# Usa una imagen base de PyTorch con herramientas de desarrollo y CUDA 11.8
FROM pytorch/pytorch:2.0.1-cu118-devel

# Metainformación
#LABEL org.opencontainers.image.source="https://github.com/FunAudioLLM/InspireMusic"
#LABEL org.opencontainers.image.licenses="Apache License 2.0"
#LABEL org.opencontainers.image.base.name="docker.io/pytorch/pytorch:2.0.1-cu118-devel"

# Establece la variable de entorno CUDA_HOME
ENV CUDA_HOME=/usr/local/cuda

# Establece las arquitecturas de CUDA para las que se compilará flash-attn
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"

# Instala dependencias básicas del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /workspace/InspireMusic

# Copia los archivos necesarios al contenedor
COPY setup.py /workspace/InspireMusic/
COPY inspiremusic /workspace/InspireMusic/inspiremusic/
COPY requirements.txt /workspace/InspireMusic/

# Instala flash-attn explícitamente
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Instala el paquete inspiremusic y sus dependencias usando setup.py
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Crea el directorio y descarga los modelos preentrenados
RUN mkdir -p /workspace/InspireMusic/pretrained_models
RUN cd /workspace/InspireMusic/pretrained_models && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-Long.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-Base-24kHz.git && \
    git clone https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz.git

# Comando por defecto para pruebas
CMD ["inspiremusic", "--help"]