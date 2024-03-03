#!/bin/bash

# Exit immediately after one of the commands failed
set -e

# Update conda
conda update -n base -c defaults conda

# Pytorch installations are machine-dependent: https://pytorch.org/
# conda install pytorch torchvision torchaudio -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

conda install pytorch-lightning -c conda-forge
# conda install pytorch-lightning pycocotools -c conda-forge
conda install pandas \
              numpy \
              ipykernel

# Update pip
pip3 install --upgrade pip

# pip3 install pytorch-lightning \
pip3 install robbytorch \
            adversarial-robustness-toolbox \
            einops