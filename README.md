# white-box-nn
This repository contains source code for the "A Conceptual Framework For White Box Neural Networks" paper.

## Recreate env

To build environment run the following commands. It's recommended to read the env_setup.sh script first.

```bash
conda create --name wbnn pip
conda activate wbnn
bash env_setup.sh
```

To recreate the experiments run the provided jupyter notebooks with the adequate kernel (wbnn conda env). Note that GPU is not required and CPU-only PyTorch installation will suffice. It should be straightforward to adapt the code to run on CUDA.