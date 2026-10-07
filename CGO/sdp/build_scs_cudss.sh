#!/bin/bash
#
# Build SCS 3.3.1 with the cuDSS (GPU direct) backend, without disturbing the
# wheel-installed scs that supplies the MKL Pardiso CPU baseline.  The result
# lands in tmp/scs-cudss; put that on PYTHONPATH to use it, and select the
# backend at runtime with linear_solver='cudss'.
#
# Three things about this box force the workarounds below:
#
#   1. This repo's path contains non-ASCII characters, which pkg-config mangles
#      into \xNN escapes, so the cuDSS tree is mirrored to /tmp/cgo-cudss.
#   2. Meson's cuda dependency wants the modern toolkit layout, i.e. a
#      targets/x86_64-linux/{include,lib} subtree.  Ubuntu's packaged toolkit
#      uses the legacy layout (headers straight into /usr/include), so that
#      subtree is assembled at /tmp/cgo-cuda and pointed at with CUDA_PATH.
#   3. The box has liblapack.so.3 but no liblapack.so, and its libblas.so.3 is
#      the reference BLAS, which carries no LAPACK.  SCS asks for lapack with
#      required=false, so without a dev symlink it links BLAS only and the
#      extensions fail at import on the undefined symbol dgeqp3_.
#
# Note the cuDSS runtime does NOT work with only libcudart on the path: with the
# system CUDA 12.0 runtime, cudssCreate returns status 5 (EXECUTION_FAILED).  It
# needs the pip 12.9 cuda_runtime/cublas/cusolver/cusparse/nvjitlink together --
# see --env below.
#
# Usage:
#   CGO/sdp/build_scs_cudss.sh          # build into tmp/scs-cudss
#   CGO/sdp/build_scs_cudss.sh --env    # print PYTHONPATH and LD_LIBRARY_PATH to use it
#
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$REPO/.venv/bin/python
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
CU12=$SITE/nvidia/cu12
WORK=$REPO/tmp
STAGE=$WORK/scs-cudss
MIRROR=/tmp/cgo-cudss
CUDA_ROOT=/tmp/cgo-cuda
BLAS_DEV=$WORK/blas-dev
TARBALL=$WORK/scs-3.3.1.tar.gz

runtime_path() {
  local path=$MIRROR/lib
  for package in cuda_runtime cublas cusolver cusparse nvjitlink cuda_nvrtc; do
    [ -d "$SITE/nvidia/$package/lib" ] && path="$path:$SITE/nvidia/$package/lib"
  done
  echo "$path"
}

if [ "${1:-}" = "--env" ]; then
  echo "PYTHONPATH=$STAGE"
  echo "LD_LIBRARY_PATH=$(runtime_path)"
  exit 0
fi

echo "==> cuDSS from pip"
PIP_CACHE_DIR=$WORK/pipcache "$PY" -m pip install --quiet nvidia-cudss-cu12

if [ ! -f "$TARBALL" ]; then
  echo "==> fetching the scs 3.3.1 sdist"
  SCS_TARBALL=$TARBALL "$PY" - <<'PYTHON'
import json, os, urllib.request
release = json.load(urllib.request.urlopen('https://pypi.org/pypi/scs/3.3.1/json', timeout=60))
url = next(f['url'] for f in release['urls'] if f['packagetype'] == 'sdist')
urllib.request.urlretrieve(url, os.environ['SCS_TARBALL'])
PYTHON
fi

echo "==> mirroring cuDSS to $MIRROR"
rm -rf "$MIRROR"
mkdir -p "$MIRROR/include" "$MIRROR/lib"
for header in "$CU12/include"/*; do ln -sfn "$header" "$MIRROR/include/"; done
for library in "$CU12/lib"/*; do ln -sfn "$library" "$MIRROR/lib/"; done
ln -sfn "$CU12/lib/libcudss.so.0" "$MIRROR/lib/libcudss.so"   # the linker wants the unsuffixed name

echo "==> pkg-config files in $WORK/pkgconfig"
mkdir -p "$WORK/pkgconfig"
cat > "$WORK/pkgconfig/cudss.pc" <<EOF
prefix=$MIRROR
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: cudss
Description: NVIDIA cuDSS (pip nvidia-cudss-cu12, mirrored to an ASCII path)
Version: 0.8.0
Libs: -L\${libdir} -lcudss
Cflags: -I\${includedir}
EOF
cat > "$WORK/pkgconfig/cuda.pc" <<EOF
prefix=/usr
includedir=\${prefix}/include
libdir=/usr/lib/x86_64-linux-gnu

Name: cuda
Description: CUDA runtime (system toolkit)
Version: 12.0
Libs: -L\${libdir} -lcudart
Cflags: -I\${includedir}
EOF

echo "==> CUDA root at $CUDA_ROOT"
target=$CUDA_ROOT/targets/x86_64-linux
rm -rf "$CUDA_ROOT"
mkdir -p "$CUDA_ROOT/bin" "$target/include" "$target/lib"
ln -sfn /usr/bin/nvcc "$CUDA_ROOT/bin/nvcc"
ln -sfn targets/x86_64-linux/include "$CUDA_ROOT/include"
ln -sfn targets/x86_64-linux/lib "$CUDA_ROOT/lib64"
for header in /usr/include/cuda* /usr/include/crt /usr/include/nvrtc* \
              /usr/include/cublas* /usr/include/cusparse* /usr/include/curand* \
              /usr/include/library_types.h /usr/include/nvJitLink.h; do
  [ -e "$header" ] && ln -sfn "$header" "$target/include/"
done
for library in /usr/lib/x86_64-linux-gnu/libcudart* /usr/lib/x86_64-linux-gnu/libcublas* \
               /usr/lib/x86_64-linux-gnu/libcusparse* /usr/lib/x86_64-linux-gnu/libcurand* \
               /usr/lib/x86_64-linux-gnu/libnvJitLink* /usr/lib/x86_64-linux-gnu/libculibos*; do
  [ -e "$library" ] && ln -sfn "$library" "$target/lib/"
done

echo "==> lapack dev symlink at $BLAS_DEV"
rm -rf "$BLAS_DEV"
mkdir -p "$BLAS_DEV"
ln -sfn /usr/lib/x86_64-linux-gnu/liblapack.so.3 "$BLAS_DEV/liblapack.so"

echo "==> building scs against cuDSS"
rm -rf "$STAGE"
cd "$WORK"
rm -rf scs-3.3.1
tar xzf "$TARBALL"
PKG_CONFIG_PATH=$WORK/pkgconfig \
LD_LIBRARY_PATH=$MIRROR/lib \
LIBRARY_PATH=$BLAS_DEV \
CUDA_PATH=$CUDA_ROOT \
PIP_CACHE_DIR=$WORK/pipcache \
  "$PY" -m pip install --target "$STAGE" --no-deps \
    -Csetup-args=-Dlink_cudss=true -Csetup-args=-Dint32=true \
    ./scs-3.3.1

echo
echo "done.  use it with:"
echo "  export PYTHONPATH=$STAGE"
echo "  export LD_LIBRARY_PATH=$(runtime_path)"
