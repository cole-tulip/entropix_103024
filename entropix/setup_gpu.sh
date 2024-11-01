#!/bin/bash

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install basic dependencies
sudo apt-get install -y \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    ninja-build \
    cmake

# Install CUDA Toolkit (11.8 is very stable with most PyTorch versions)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override

# Add CUDA paths to bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip3 install tiktoken
pip3 install tyro
pip3 install jax jaxlib-cuda11x -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install ml_dtypes
pip3 install numpy

# Verify installations
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create project directories
mkdir -p weights/1B-Instruct
mkdir -p entropix