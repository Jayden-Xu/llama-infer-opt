#!/usr/bin/env bash
# ABB-OPT: one-shot configure + build for the inference-opt workflow.
#
# Detects the platform (CUDA / Metal / CPU), picks sensible cmake flags,
# and builds the targets we actually exercise in this repo (llama-server +
# llama-mtmd-cli). Re-running the script is fine -- cmake reuses the build
# directory, only changed files recompile.
#
# Usage:
#   scripts/build.sh                        # auto-detect backend
#   BACKEND=cuda scripts/build.sh           # force CUDA
#   BACKEND=metal scripts/build.sh          # force Metal (macOS)
#   BACKEND=cpu scripts/build.sh            # CPU only
#   CUDA_ARCH=86 scripts/build.sh           # 3060 / 3050 / Orin (sm_86 / sm_87)
#   CUDA_ARCH=89 scripts/build.sh           # 4070 (sm_89)
#   BUILD_DIR=build-cuda scripts/build.sh   # custom build dir
#   JOBS=4 scripts/build.sh                 # limit parallelism
#   TARGETS="llama-server" scripts/build.sh # override targets

set -euo pipefail

LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LLAMA_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
TARGETS="${TARGETS:-llama-server llama-mtmd-cli}"

# ----- backend detection ----------------------------------------------
if [[ -z "${BACKEND:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        BACKEND=cuda
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        BACKEND=metal
    else
        BACKEND=cpu
    fi
fi

# ----- cuda arch detection (3060 / 3050 / Orin -> 86, 4070 -> 89, etc.)
if [[ "$BACKEND" == "cuda" ]]; then
    if [[ -z "${CUDA_ARCH:-}" ]]; then
        # try to read SM from nvidia-smi (e.g. "8.6")
        if command -v nvidia-smi >/dev/null 2>&1; then
            sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
            if [[ -n "$sm" && "$sm" =~ ^[0-9]+$ ]]; then
                CUDA_ARCH="$sm"
            fi
        fi
        CUDA_ARCH="${CUDA_ARCH:-86}"
    fi
fi

# ----- cmake configure ------------------------------------------------
GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR=(-G Ninja)
fi

cmake_flags=(-DCMAKE_BUILD_TYPE=Release)
case "$BACKEND" in
    cuda)
        cmake_flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
        echo "[build] backend=cuda  CUDA_ARCH=$CUDA_ARCH"
        ;;
    metal)
        cmake_flags+=(-DGGML_METAL=ON)
        echo "[build] backend=metal"
        ;;
    cpu)
        cmake_flags+=(-DGGML_CUDA=OFF -DGGML_METAL=OFF)
        echo "[build] backend=cpu"
        ;;
    *)
        echo "[build] unknown BACKEND=$BACKEND (supported: cuda metal cpu)" >&2
        exit 1
        ;;
esac

echo "[build] cmake -B $BUILD_DIR ${cmake_flags[*]} ${GENERATOR[*]:-}"
cmake -B "$BUILD_DIR" "${cmake_flags[@]}" "${GENERATOR[@]}"

# ----- build -----------------------------------------------------------
echo "[build] cmake --build $BUILD_DIR -j$JOBS --target $TARGETS"
# shellcheck disable=SC2086
cmake --build "$BUILD_DIR" -j"$JOBS" --target $TARGETS

echo
echo "[build] done. binaries:"
for t in $TARGETS; do
    bin="$BUILD_DIR/bin/$t"
    [[ -x "$bin" ]] && ls -lh "$bin" || echo "  (missing) $bin"
done
