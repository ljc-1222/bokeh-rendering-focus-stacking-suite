"""Build script for the `scatter_cuda` PyTorch C++/CUDA extension.

This file is invoked by:
- `scripts/build_scatter_cuda.sh`
- `setup.sh` (during environment provisioning)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="scatter_cuda",
    ext_modules=[
        CUDAExtension(
            "scatter_cuda",
            [
                "scatter_cuda.cpp",
                "scatter_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
