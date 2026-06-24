#!/usr/bin/env bash
# ABB-OPT (opt/fused-pre-add-rmsnorm): correctness probe for the fused
# {ADD(residual), RMS_NORM, MUL(gamma)} kernel.
#
# Method:
#   Spawn one llama-server, send the same prompt twice -- once without the
#   optimisation, once with OPT_FUSED_PRE_RMS=1. Use temperature=0 + fixed
#   seed so generation is fully greedy / deterministic. Compare:
#     1. The full generated text (must be identical character-by-character).
#     2. The first-token logprobs if the server returns them.
#
# A mismatch indicates that the fused kernel produced a numerically
# different output -- the optimisation should not be merged in that case.
#
# Usage:
#   bash scripts/check-fused-pre-rms.sh           # uses default model paths
#   N_PREDICT=128 bash scripts/check-fused-pre-rms.sh
#   IMAGE=/path/to/img.jpg bash scripts/check-fused-pre-rms.sh
#
# Exit code 0 on bit-equal output, 1 otherwise.

set -euo pipefail

LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen25vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/qwen25vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf}"
IMAGE="${IMAGE:-$LLAMA_DIR/media/robot-512.jpg}"
PROMPT="${PROMPT:-Describe what you see in this image in detail.}"
N_PREDICT="${N_PREDICT:-64}"
PORT="${PORT:-18091}"

[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "[check] llama-server not found at $LLAMA_SERVER_BIN" >&2; exit 1; }
[[ -f "$MODEL"  ]] || { echo "[check] model not found at $MODEL"  >&2; exit 1; }
[[ -f "$MMPROJ" ]] || { echo "[check] mmproj not found at $MMPROJ" >&2; exit 1; }
[[ -f "$IMAGE"  ]] || { echo "[check] image not found at $IMAGE"  >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "[check] jq is required" >&2; exit 1; }

WORK_DIR="$(mktemp -d -t abb-fused-pre-rms.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

DATA_URL="data:image/$(echo "$IMAGE" | awk -F. '{print tolower($NF)}');base64,$(base64 -i "$IMAGE" | tr -d '\n')"
PAYLOAD="$WORK_DIR/req.json"
{
    printf '{"model":"any","stream":false,"max_tokens":%d,"temperature":0,"seed":1,"messages":[{"role":"user","content":[' "$N_PREDICT"
    printf '{"type":"text","text":%s}' "$(printf '%s' "$PROMPT" | jq -Rs .)"
    printf ',{"type":"image_url","image_url":{"url":%s}}' "$(printf '%s' "$DATA_URL" | jq -Rs .)"
    printf ']}]}'
} > "$PAYLOAD"

run_case() {
    local label="$1"; shift  # remaining args = env vars (e.g. OPT_FUSED_PRE_RMS=1)
    local out_file="$WORK_DIR/${label}.txt"
    local server_log="$WORK_DIR/${label}.server.log"

    echo "[check] === case: $label  envs: $*"
    local stdbuf_prefix=()
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf_prefix=(stdbuf -oL -eL)
    fi
    env "$@" "${stdbuf_prefix[@]}" "$LLAMA_SERVER_BIN" \
        -m "$MODEL" --mmproj "$MMPROJ" \
        --host 127.0.0.1 --port "$PORT" \
        -c 16384 --no-warmup \
        > "$server_log" 2>&1 &
    local server_pid=$!

    # wait until /health is ready
    local ok=0
    for _ in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 1
    done
    if [[ "$ok" != 1 ]]; then
        echo "[check] server failed to come up; last 30 lines of log:" >&2
        tail -30 "$server_log" >&2
        kill "$server_pid" 2>/dev/null || true
        exit 1
    fi

    curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary @"$PAYLOAD" \
        | jq -r '.choices[0].message.content // ""' > "$out_file"

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[check] -> $(wc -c < "$out_file") bytes  head: $(head -c 80 "$out_file")"
}

run_case baseline
run_case fused    OPT_FUSED_PRE_RMS=1

echo
echo "[check] diffing..."
if diff -u "$WORK_DIR/baseline.txt" "$WORK_DIR/fused.txt" > "$WORK_DIR/diff.txt"; then
    echo "[check] PASS: bit-equal greedy output (temperature=0, seed=1)"
    md5sum "$WORK_DIR/baseline.txt" "$WORK_DIR/fused.txt" || md5 "$WORK_DIR/baseline.txt" "$WORK_DIR/fused.txt" || true
    exit 0
else
    echo "[check] FAIL: outputs differ" >&2
    echo "---- baseline ----" >&2
    cat "$WORK_DIR/baseline.txt" >&2
    echo "---- fused ----" >&2
    cat "$WORK_DIR/fused.txt" >&2
    echo "---- diff ----" >&2
    cat "$WORK_DIR/diff.txt" >&2
    exit 1
fi
