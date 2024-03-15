# white-box-nn
This is a source code for the paper "A Conceptual Framework For White Box Neural Networks".

## Recreate env

To build environment run the following commands. It's recommended to read the env_setup.sh script first.

```bash
conda create --name wbnn pip
conda activate wbnn
bash env_setup.sh
```

To recreate the experiments run the provided jupyter notebooks with the adequate kernel. Note that GPU is not required and CPU-only PyTorch installation will suffice. In fact the code wasn't even tested on GPU (however it should be straightforward to adapt it to run on CUDA if you feel the need for it).