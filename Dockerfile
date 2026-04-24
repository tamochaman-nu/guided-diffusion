FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-mpi4py \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install PyTorch with CUDA 11.8 support before other dependencies
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
RUN pip install -e .

# Install additional dependencies required for scripts and evaluation
RUN pip install scipy requests pillow

# Note: tensorflow-gpu is not installed by default because it is deprecated and
# may fail to install on newer Python/runtime combinations. Install it manually
# only if you need TensorFlow evaluation support.

# Default command
CMD ["bash"]