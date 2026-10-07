"""Python side of the CUDA extension of ``CGO/gaussian.py``.  The kernel lives in
``CGO/cuda/gaussian.cu`` and is built just in time with
``torch.utils.cpp_extension.load``; the compiled module is cached in
``CGO/cuda/build``, so only the first import pays for the compilation.  The first
build needs ninja (``python -m pip install ninja``); later imports load the
cached ``CGO/cuda/build/gaussian_ext.so`` directly.

The extension exposes one object, ``PatchProjector``, which derives and owns the
register geometry of a patch; this module only loads it.
"""
import importlib.util
import os
import sys

import torch

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


# The extension class, built from (rotation, is_mixed); its constructor, `vacuum`
# and `apply` are documented in CGO/cuda/gaussian.cu.
PatchProjector = _extension.PatchProjector
