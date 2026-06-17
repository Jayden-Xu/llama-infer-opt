#!/usr/bin/env bash
# ABB-OPT: standalone ViT-cache correctness check.
#
# Boots llama-server with OPT_VIT_CACHE=1 and MTMD_DUMP_EMBD=<path>, sends
# the same single-image request twice with a varying text prefix to defeat
# the LLM prompt cache (so the server actually calls mtmd_batch_encode each
# time). The first request misses and writes <path>.000 from the forward
# path; the second request hits and writes <path>.001 from the cache path.
# Then runs scripts/diff-embd.py to compare the two dumps. A correct cache
# should produce max|diff|=0 / cos_sim=1.

set -euo pipefail

LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LLAMA_DIR"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen25vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/qwen25vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf}"
IMAGE="${IMAGE:-$LLAMA_DIR/media/robot-512.jpg}"
PORT="${PORT:-18091}"
DUMP_DIR="${DUMP_DIR:-/tmp/vit-cache-check}"

[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "missing $LLAMA_SERVER_BIN"; exit 1; }
[[ -f "$MODEL"            ]] || { echo "missing $MODEL"; exit 1; }
[[ -f "$MMPROJ"           ]] || { echo "missing $MMPROJ"; exit 1; }
[[ -f "$IMAGE"            ]] || { echo "missing $IMAGE"; exit 1; }
command -v jq      >/dev/null || { echo "need jq";      exit 1; }
command -v python3 >/dev/null || { echo "need python3"; exit 1; }

mkdir -p "$DUMP_DIR"
rm -f "$DUMP_DIR"/*

echo "[check] booting server with OPT_VIT_CACHE=1, MTMD_DUMP_EMBD=$DUMP_DIR/run"
OPT_VIT_CACHE=1 MTMD_DUMP_EMBD="$DUMP_DIR/run" \
    "$LLAMA_SERVER_BIN" \
        -m "$MODEL" --mmproj "$MMPROJ" \
        --host 127.0.0.1 --port "$PORT" \
        -c 16384 --no-warmup -v \
        > "$DUMP_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# wait for /health
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done

DATA_URL="data:image/$(echo "$IMAGE" | awk -F. '{print tolower($NF)}');base64,$(base64 -w0 "$IMAGE")"

build_payload() {
    local tag="$1"; local out="$2"
    {
        printf '{"model":"any","stream":false,"max_tokens":1,"temperature":0,"seed":1,"messages":[{"role":"user","content":['
        printf '{"type":"text","text":%s}' "$(printf '[%s] hi' "$tag" | jq -Rs .)"
        printf ',{"type":"image_url","image_url":{"url":%s}}' "$(printf '%s' "$DATA_URL" | jq -Rs .)"
        printf ']}]}'
    } > "$out"
}

build_payload run-A "$DUMP_DIR/req-A.json"
build_payload run-B "$DUMP_DIR/req-B.json"

echo "[check] sending request A (cache miss expected)"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @"$DUMP_DIR/req-A.json" >/dev/null

echo "[check] sending request B (cache hit expected)"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @"$DUMP_DIR/req-B.json" >/dev/null

# give the server a moment to flush buffered log lines
sleep 0.5
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
trap - EXIT

echo
echo "[check] dumps:"
ls -la "$DUMP_DIR"/run.* 2>/dev/null || { echo "no dumps produced"; exit 1; }

echo
echo "[check] cache hits / misses summary from server log:"
grep -E "ViT embd cache (hit|miss)" "$DUMP_DIR/server.log" || true

echo
if [[ -f "$DUMP_DIR/run.000" && -f "$DUMP_DIR/run.001" ]]; then
    echo "[check] byte-level diff (md5 + cmp) of run.000 (forward) vs run.001 (cache hit)"
    md5_a=$(md5sum "$DUMP_DIR/run.000" | awk '{print $1}')
    md5_b=$(md5sum "$DUMP_DIR/run.001" | awk '{print $1}')
    echo "  run.000 md5 : $md5_a"
    echo "  run.001 md5 : $md5_b"
    if cmp -s "$DUMP_DIR/run.000" "$DUMP_DIR/run.001"; then
        echo "  cmp         : IDENTICAL (every byte matches)"
    else
        echo "  cmp         : DIFFER -- first 10 byte positions:"
        cmp -l "$DUMP_DIR/run.000" "$DUMP_DIR/run.001" | head -10
    fi

    echo
    echo "[check] numeric diff (numpy) of the float32 payload"
    python3 scripts/diff-embd.py "$DUMP_DIR/run.000" "$DUMP_DIR/run.001"
else
    echo "[check] expected exactly run.000 and run.001 dumps, got something else"
    exit 1
fi
