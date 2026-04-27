#!/bin/bash
set -e

conda create -y --name aurafusion360 python=3.10
conda activate aurafusion360
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
# module load cuda/12.4
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y ffmpeg=4.2.2
conda install -y typing_extensions=4.9.0
pip3 install torch torchvision torchaudio


pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/sam2.git'
