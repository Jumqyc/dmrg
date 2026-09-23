"""Python side of the CUDA extension of the free-fermion contraction helpers of
``FreeFermion/linalg.py``.  The kernels live in ``FreeFermion/cuda/fermion.cu``
and are built just in time with ``torch.utils.cpp_extension.load``; the compiled
module is cached in ``FreeFermion/cuda/build``, so only the first import pays for
the compilation.  The first build needs ninja (``python -m pip install ninja``);
later imports load the cached ``FreeFermion/cuda/build/fermion_ext.so`` directly.
"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')
LIBRARY = os.path.join(BUILD_DIR, 'fermion_ext.so')

_extensions = os.environ.get('TORCH_EXTENSIONS_DIR')
if _extensions is None:
    os.environ['TORCH_EXTENSIONS_DIR'] = BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

if os.path.exists(LIBRARY):
    _spec = importlib.util.spec_from_file_location('fermion_ext', LIBRARY)
    _extension = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_extension)
else:
    # the build needs ninja on PATH, which a bare .venv/bin/python does not add
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    _extension = load(name='fermion_ext',
                      sources=[os.path.join(_HERE, 'fermion.cu')],
                      extra_cuda_cflags=['-O3'],
                      build_directory=BUILD_DIR,
                      verbose=False)


def operator_vectors(coeff: Tensor, codes: Tensor) -> Tensor:
    '''
    Logic: batched mode-basis vectors of a list of operators (FreeFermion/cuda/fermion.cu):
    kind 0 gives the unit vector of d_k, kind 1 the unit vector of d_k^dag and
    kind 2 the inverse expansion of the physical Majorana gamma_mu.
    Args:
        coeff: complex Williamson coefficients of d_k over the 2n Majoranas,
            shape (n, 2n).
        codes: integer operator codes, shape (..., m, 2), with the kind in
            codes[..., 0] (0 = d_k, 1 = d_k^dag, 2 = gamma_mu) and the index in
            codes[..., 1]; every code has to be valid, there is no padding.
    Returns:
        The complex vectors of the m operators, shape (..., m, 2n).
    '''
    return _extension.operator_vectors(coeff, codes)


def vacuum_contraction(vectors: Tensor, num_modes: int) -> Tensor:
    '''
    Logic: batched Wick contraction matrices K_ij = <0|A_i A_j|0> (FreeFermion/cuda/fermion.cu):
    the sum of the first half of a_i with the second half of a_j, upper triangle
    kept and antisymmetrised.
    Args:
        vectors: the m operators as mode-basis vectors, shape (..., m, 2n).
        num_modes: number of modes n, the split point of the two halves.
    Returns:
        The complex antisymmetric contraction matrices, shape (..., m, m).
    '''
    return _extension.vacuum_contraction(vectors, num_modes)


def pfaffian(matrices: Tensor) -> Tensor:
    '''
    Logic: batched Pfaffian by skew-symmetric elimination with two-by-two pivots
    (FreeFermion/cuda/fermion.cu), one block per matrix; a matrix without a pivot candidate
    gives 0, an odd size gives 0 and size 0 gives 1.
    Args:
        matrices: real or complex antisymmetric matrices, shape (..., m, m) with at
            least one batch dimension.
    Returns:
        The scalar tensors sum over pairings of the signed products of the
        entries, shape (...,).
    '''
    return _extension.pfaffian(matrices)
