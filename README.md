# white-box-nn
This repository contains source code for the [A Conceptual Framework For White Box Neural Networks](https://arxiv.org/abs/2403.09863) paper.

## Recreate env

To build environment run the following commands. It's recommended to read the env_setup.sh script first. In particular PyTorch/CUDA installation should be adjusted for your machine (however CPU-only PyTorch installation will suffice).

```bash
conda create --name wbnn pip
conda activate wbnn
bash env_setup.sh
```

To recreate the experiments run the provided jupyter notebooks with the adequate kernel (wbnn conda env).