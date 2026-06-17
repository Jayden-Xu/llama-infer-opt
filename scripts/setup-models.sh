#!/usr/bin/env bash
# ABB-OPT: download VLM gguf models used by the bench harness.
#
# Usage:
#   scripts/setup-models.sh                   # download everything missing
#   MODELS_DIR=~/m scripts/setup-models.sh    # custom destination
#   MODELS=qwen25vl scripts/setup-models.sh   # subset (space-separated)
#
# Default destination matches bench-vlm-server.sh defaults so no extra flags
# are needed when running the bench afterwards.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODELS="${MODELS:-qwen25vl}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[setup_models] python3 is required" >&2
    exit 1
fi
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "[setup_models] installing huggingface_hub..."
    python3 -m pip install --user --quiet huggingface_hub
}

echo "[setup_models] dest=$MODELS_DIR  models=$MODELS"
mkdir -p "$MODELS_DIR"

dl() {
    local sub="$1"; local repo="$2"; shift 2
    local dest="$MODELS_DIR/$sub"
    mkdir -p "$dest"
    local patterns=("$@")
    echo "[setup_models] -> $sub  ($repo)"
    python3 - "$repo" "$dest" "${patterns[@]}" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, dest, *patterns = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    allow_patterns=list(patterns),
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print(f"  done -> {dest}")
PY
}

for m in $MODELS; do
    case "$m" in
        qwen25vl)
            dl qwen25vl "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF" "*Q4_K_M*" "mmproj*"
            ;;
        *)
            echo "[setup_models] unknown model: $m (supported: qwen25vl)" >&2
            exit 1
            ;;
    esac
done

# Normalise mmproj filenames so bench-vlm-server.sh defaults work as-is.
# bench expects: mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
QWEN_MMPROJ_EXPECTED="$MODELS_DIR/qwen25vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
if [[ ! -f "$QWEN_MMPROJ_EXPECTED" ]]; then
    actual=$(ls "$MODELS_DIR/qwen25vl/"mmproj*.gguf 2>/dev/null | head -1 || true)
    if [[ -n "$actual" ]]; then
        ln -sf "$(basename "$actual")" "$QWEN_MMPROJ_EXPECTED"
        echo "[setup_models] symlinked $(basename "$actual") -> mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
    fi
fi

echo
echo "[setup_models] ls $MODELS_DIR:"
find "$MODELS_DIR" -maxdepth 2 -type f -name "*.gguf" -exec ls -lh {} \;
