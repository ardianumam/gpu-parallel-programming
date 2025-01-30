from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polynomial_cuda',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('polynomial_cuda', [
            'polynomial_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })