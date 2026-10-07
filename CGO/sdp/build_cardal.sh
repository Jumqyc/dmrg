#!/bin/bash
#
# Build CARDAL (the GPU low-rank SDP solver) on this machine.  `pip install
# cardal` alone does not work here; this is what it takes:
#
#   1. CARDAL calls cuSPARSE's cusparseSpMV_preprocess, which the system toolkit
#      does not have.  nvcc here is Ubuntu's 12.0 (/usr/bin/nvcc -> nvidia-cuda-
#      toolkit 12.0), the symbol appeared after 12.0.  The pip nvidia-*-cu12
#      packages supply 12.9 headers and host libraries, and nvcc 12.0 compiles
#      against them fine.
#   2. The nvcc frontend and the DEVICE runtime must stay at 12.0: nvlink 12.0
#      refuses a libcudadevrt from 12.9 ("newer than toolkit 129 vs 120").  So
#      cuda_runtime must not appear in the nvcc -L flags; it only belongs to
#      LD_LIBRARY_PATH, where the loader needs nvJitLink 12.9 for cuSPARSE 12.9.
#   3. nvcc 12.0 rejects gcc 13, so the host compiler has to be gcc-12.
#
# Usage:
#   CGO/sdp/build_cardal.sh          # build and install cardal
#   CGO/sdp/build_cardal.sh --env    # print the LD_LIBRARY_PATH cardal needs at import
#
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$REPO/.venv/bin/python
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
NV=$SITE/nvidia
MERGED=$REPO/tmp/cardal-cuda          # inside the gitignored tmp/

CUDA_PACKAGES=(
  nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
  nvidia-cusparse-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12
)

# The pip CUDA libraries cardal resolves at import time.
runtime_path() {
  local ld=$MERGED/lib64
  for package in cusparse cublas cusolver nvjitlink cuda_nvrtc cuda_runtime; do
    [ -d "$NV/$package/lib" ] && ld="$ld:$NV/$package/lib"
  done
  echo "$ld"
}

if [ "${1:-}" = "--env" ]; then
  echo "LD_LIBRARY_PATH=$(runtime_path)"
  exit 0
fi

for compiler in gcc-12 g++-12; do
  command -v "$compiler" >/dev/null || { echo "missing $compiler (nvcc 12.0 rejects gcc 13)" >&2; exit 1; }
done

echo "==> pip CUDA 12.9 components"
PIP_CACHE_DIR=$REPO/tmp/pipcache "$PY" -m pip install --quiet "${CUDA_PACKAGES[@]}"

echo "==> merged CUDA tree at $MERGED"
rm -rf "$MERGED"
mkdir -p "$MERGED/bin" "$MERGED/include" "$MERGED/lib64"

includes=""
for package in cusparse cublas cusolver; do
  [ -d "$NV/$package/include" ] || continue
  includes="$includes -I$NV/$package/include"
  for header in "$NV/$package/include"/*; do ln -sfn "$header" "$MERGED/include/"; done
done

# Only the host libraries go into nvcc's -L; see (2) above.
host_libs=""
for package in cusparse cublas cusolver nvjitlink; do
  [ -d "$NV/$package/lib" ] || continue
  for library in "$NV/$package/lib"/*; do ln -sfn "$library" "$MERGED/lib64/"; done
  host_libs="$host_libs -L$NV/$package/lib"
done
for package in cuda_runtime cuda_nvrtc cuda_nvcc; do
  [ -d "$NV/$package/lib" ] || continue
  for library in "$NV/$package/lib"/*; do ln -sfn "$library" "$MERGED/lib64/"; done
done
ln -sfn "$NV/cuda_nvcc/nvvm" "$MERGED/nvvm"

printf '#!/bin/sh\nexec /usr/bin/nvcc%s%s "$@"\n' "$includes" "$host_libs" > "$MERGED/bin/nvcc"
chmod +x "$MERGED/bin/nvcc"

echo "==> building cardal"
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
CUDACXX=$MERGED/bin/nvcc \
CUDA_PATH=$MERGED \
PATH=$MERGED/bin:$PATH \
LIBRARY_PATH=$(runtime_path) \
LD_LIBRARY_PATH=$(runtime_path) \
PIP_CACHE_DIR=$REPO/tmp/pipcache \
SKBUILD_CMAKE_ARGS="-DCUDAToolkit_ROOT=$MERGED -DCMAKE_CUDA_COMPILER=$MERGED/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_MATIO=OFF" \
  "$PY" -m pip install cardal

echo
echo "done.  cardal needs these libraries at import time:"
echo "  export LD_LIBRARY_PATH=\$(CGO/build_cardal.sh --env | cut -d= -f2-)"
