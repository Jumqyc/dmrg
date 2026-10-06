"""Python side of the CUDA extension of ``CGO/gaussian.py``.  The kernel lives in
``CGO/cuda/gaussian.cu`` and is built just in time with
``torch.utils.cpp_extension.load``; the compiled module is cached in
``CGO/cuda/build``, so only the first import pays for the compilation.  The first
build needs ninja (``python -m pip install ninja``); later imports load the
cached ``CGO/cuda/build/gaussian_ext.so`` directly.
"""
import importlib.util
import os
import sys

import torch

from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')
LIBRARY = os.path.join(BUILD_DIR, 'gaussian_ext.so')

if os.environ.get('TORCH_EXTENSIONS_DIR') is None:
    os.environ['TORCH_EXTENSIONS_DIR'] = BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

if os.path.exists(LIBRARY):
    _spec = importlib.util.spec_from_file_location('gaussian_ext', LIBRARY)
    _extension = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_extension)
else:
    # the build needs ninja on PATH, which a bare .venv/bin/python does not add
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    _extension = load(name='gaussian_ext',
                      sources=[os.path.join(_HERE, 'gaussian.cu')],
                      extra_cuda_cflags=['-O3'],
                      build_directory=BUILD_DIR,
                      verbose=False)


def apply_label(amps: Tensor,
                elem_even: Tensor,
                elem_odd: Tensor,
                below: Tensor,
                flip_mask: Tensor,
                subset_flip: Tensor,
                pure_lo: Tensor,
                pure_hi: Tensor,
                mixed: Tensor) -> Tensor:
    '''
    Apply one Majorana label, expanded over the normal modes, to a whole batch of
    patch states (CGO/cuda/gaussian.cu).
    Args:
        amps: the amplitude state, shape (pure_count, size, size), complex128 on CUDA.
        elem_even: the even-Majorana rotation column of the label, shape (n_modes,),
            complex128 on CUDA.
        elem_odd: the odd-Majorana rotation column of the label, shape (n_modes,).
        below: mixed-register mask of the modes below each mode, int32.
        flip_mask: mixed-register bit each mode flips, int32, 0 for a pure mode.
        subset_flip: pure-register bit each mode flips, int32, 0 for a mixed mode.
        pure_lo: pure-register mask below each mode, int32.
        pure_hi: `pure_lo` plus the mode's own pure bit, int32.
        mixed: 1 for a mixed mode, int8.
    Returns:
        The state after the label, shape (pure_count, size, size).
    '''
    return _extension.apply_label(amps, elem_even, elem_odd, below, flip_mask,
                                  subset_flip, pure_lo, pure_hi, mixed)
