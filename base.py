"""Shared constants of the repository: the default device the tensor-network code
runs on, and the complex and real dtypes it is written in.
"""
import torch

# Default device and complex dtype of the library.
CUDA = torch.device('cuda')
COMPLEX = torch.complex128
REAL = torch.float64
