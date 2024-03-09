#!/bin/bash

# Exit immediately after one of the commands failed
set -e

# Update conda
conda update -n base -c defaults conda

# Pytorch installations are machine-dependent: https://pytorch.org/
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install pytorch-lightning -c conda-forge

conda install pandas \
              numpy \
              ipykernel

# Update pip
pip3 install --upgrade pip

# pip3 install pytorch-lightning \
pip3 install robbytorch \
            adversarial-robustness-toolbox \
            multiprocess \  # to fix adversarial-robustness-toolbox error
            einops \
            kornia