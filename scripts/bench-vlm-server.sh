#!/usr/bin/env bash
# ABB-OPT: VLM multi-image bench harness driving llama-server.
#
# Why this exists:
#   bench-vlm.sh drives llama-mtmd-cli, which iterates chunks via
#   mtmd_helper_eval_chunks (one image per encode call). It cannot exercise
#   the qwen-vl multi-image ViT batch-prefill path (n_img_batch > 1) added
#   under opt/vit-batch-prefill -- that path only triggers via
#   mtmd_batch_encode, which llama-server uses (server-context.cpp).
#
# What this does:
#   1. Boots llama-server with the given mmproj.
#   2. Sends N_REQUESTS chat/completions requests, each containing N_IMAGES
#      copies of the same image (base64-encoded data URL, OpenAI vision
#      format). The server accumulates the media chunks and calls
#      mtmd_batch_encode, packing all N images into one ViT forward.
#   3. Parses the response timings (prompt eval ms, total ms) and the
#      ViT encode time from the server stderr log ("image slice encoded in
#      X ms" emitted by clip_image_batch_encode).
#   4. Aggregates mean / stddev / min / max per case, prints a comparison
#      table vs baseline.
#   5. Writes raw logs + summaries + comparison.txt under bench-results/.
#
# Usage:
#   scripts/bench-vlm-server.sh                    # N_RUNS=10, N_IMAGES=3, N_TURNS=3
#   scripts/bench-vlm-server.sh 20                 # N_RUNS=20
#   N_IMAGES=3 scripts/bench-vlm-server.sh         # 3 images per request (multi-view)
#   N_TURNS=5 scripts/bench-vlm-server.sh          # 5 turns of dialogue per session
#   SKIP_BUILD=1 scripts/bench-vlm-server.sh       # reuse existing binary
#
# Session model:
#   one "run" = one session = N_TURNS sequential requests, all carrying the
#   same N_IMAGES images but different text prefixes ([run=...-turn=...]).
#   First turn populates the ViT cache (N_IMAGES misses); subsequent turns
#   should all hit. Per-session totals are summed across turns so the bench
#   reports per-session ViT_E2E / prompt_eval / VLM_E2E -- the number a
#   product owner cares about.
#
# Env knobs (defaults work for Mac dev box; change for 4070 etc.):
#   LLAMA_SERVER_BIN  default: <repo>/build/bin/llama-server
#   MODEL             default: ~/models/qwen25vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
#   MMPROJ            default: ~/models/qwen25vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
#   IMAGE             default: <repo>/media/robot-512.jpg
#   PROMPT            default: "Describe what you see across these images."
#   N_PREDICT         default: 64
#   N_IMAGES          number of images packed per request (default: 3)
#   N_TURNS           number of dialogue turns per session     (default: 3)
#   PORT              default: 18080
#   ENV_TAG           autodetected (mac-metal | cuda-<gpu> | linux)
#
# Workflow (per branch):
#   - dev/infer-opt: only "baseline" line in CASES (multi-image runs through
#     unchanged code -> serial fallback because n_img_batch = 1).
#   - opt/vit-batch-prefill: append a "vit-batch" line with no extra OPT_*
#     because the batch path is unconditional once N_IMAGES > n_temporal_merge.
#     The interesting comparison here is dev/infer-opt vs opt/vit-batch-prefill,
#     not env-gated cases on the same branch.

set -euo pipefail

# ----- cases: "<label> <env-var-list>" ----------------------------------
# NOTE: OPT_FUSED_PRE_RMS is parked; it produces incorrect output because
# ggml-alloc may alias the ADD node's buffer with other tensors. Re-enable
# only after the ggml-sched / alloc work that lets a fused op produce two
# distinct outputs lands.
CASES=(
    "baseline                "                       # nothing enabled
)

# ----- config ----------------------------------------------------------
N_RUNS="${1:-${N_RUNS:-10}}"
N_IMAGES="${N_IMAGES:-3}"
N_TURNS="${N_TURNS:-3}"
PORT="${PORT:-18080}"

LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_DIR="$(cd "$LLAMA_DIR/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen25vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/qwen25vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf}"

IMAGE_DEFAULT="$LLAMA_DIR/media/robot-512.jpg"
IMAGE_URL_DEFAULT="https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=512&h=512&fit=crop"
IMAGE="${IMAGE:-$IMAGE_DEFAULT}"
PROMPT="${PROMPT:-Describe what you see across these images.}"
N_PREDICT="${N_PREDICT:-64}"

if [[ "$IMAGE" == "$IMAGE_DEFAULT" && ! -f "$IMAGE" ]]; then
    echo "[bench_vlm_server] default image not found, downloading 512x512 robot scene..."
    if   command -v wget >/dev/null 2>&1; then wget -q -O "$IMAGE" "$IMAGE_URL_DEFAULT"
    elif command -v curl >/dev/null 2>&1; then curl -sSL -o "$IMAGE" "$IMAGE_URL_DEFAULT"
    else echo "[bench_vlm_server] need wget or curl to fetch default image" >&2; exit 1
    fi
    [[ -s "$IMAGE" ]] || { echo "[bench_vlm_server] download failed" >&2; exit 1; }
fi
[[ -f "$IMAGE" ]] || { echo "[bench_vlm_server] image not found at $IMAGE" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "[bench_vlm_server] model not found at $MODEL" >&2; exit 1; }
[[ -f "$MMPROJ" ]] || { echo "[bench_vlm_server] mmproj not found at $MMPROJ" >&2; exit 1; }

if ! command -v jq >/dev/null 2>&1; then
    echo "[bench_vlm_server] jq is required to parse server JSON; please install jq" >&2
    exit 1
fi

if [[ -z "${ENV_TAG:-}" ]]; then
    case "$(uname)" in
        Darwin) ENV_TAG="mac-metal" ;;
        Linux)
            if command -v nvidia-smi >/dev/null 2>&1; then
                gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '-' | tr -d '()')
                ENV_TAG="cuda-${gpu:-unknown}"
            else
                ENV_TAG="linux"
            fi
            ;;
        *) ENV_TAG="$(uname | tr '[:upper:]' '[:lower:]')" ;;
    esac
fi

DATE_TAG="$(date +%Y-%m-%d)"
OUT_DIR="$ROOT_DIR/bench-results"
mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/${DATE_TAG}-${ENV_TAG}-vlm-server-bench"
mkdir -p "$RUN_DIR"

echo "[bench_vlm_server] env=$ENV_TAG  N_RUNS=$N_RUNS  N_IMAGES=$N_IMAGES  N_TURNS=$N_TURNS  out=$RUN_DIR"
echo "[bench_vlm_server] server=$LLAMA_SERVER_BIN  port=$PORT"
echo "[bench_vlm_server] image=$IMAGE  prompt=\"$PROMPT\"  n_predict=$N_PREDICT"
echo "[bench_vlm_server] cases:"
for c in "${CASES[@]}"; do echo "    - $c"; done
echo

# ----- build -----------------------------------------------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    CMAKE_BIN="${CMAKE_BIN:-cmake}"
    if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
        if   [[ -x /opt/homebrew/bin/cmake ]]; then CMAKE_BIN=/opt/homebrew/bin/cmake
        elif [[ -x /usr/local/bin/cmake    ]]; then CMAKE_BIN=/usr/local/bin/cmake
        else echo "[bench_vlm_server] cmake not found, set CMAKE_BIN or SKIP_BUILD=1" >&2; exit 1
        fi
    fi
    BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build}"
    JOBS="${JOBS:-8}"
    echo "[bench_vlm_server] build: $CMAKE_BIN --build $BUILD_DIR -j$JOBS --target llama-server"
    "$CMAKE_BIN" --build "$BUILD_DIR" -j"$JOBS" --target llama-server
    echo
fi

[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "[bench_vlm_server] server bin not found at $LLAMA_SERVER_BIN" >&2; exit 1; }

# ----- payload --------------------------------------------------------
# Build the JSON request body: N_IMAGES copies of the same image in a single
# chat completions request. The text prompt is parameterised so we can vary
# it per-run and defeat server prompt-cache (otherwise runs 2..N would skip
# the image encode entirely and we'd lose all ViT timings).
echo "[bench_vlm_server] preparing request payload template (N_IMAGES=$N_IMAGES)..."
DATA_URL="data:image/$(echo "$IMAGE" | awk -F. '{print tolower($NF)}');base64,$(base64 -i "$IMAGE" | tr -d '\n')"

build_payload() {
    local prompt_text="$1"; local out="$2"
    {
        printf '{"model":"any","stream":false,"max_tokens":%d,"temperature":0,"seed":1,"messages":[{"role":"user","content":[' "$N_PREDICT"
        printf '{"type":"text","text":%s}' "$(printf '%s' "$prompt_text" | jq -Rs .)"
        for i in $(seq 1 "$N_IMAGES"); do
            printf ',{"type":"image_url","image_url":{"url":%s}}' "$(printf '%s' "$DATA_URL" | jq -Rs .)"
        done
        printf ']}]}'
    } > "$out"
}

# ----- runner ----------------------------------------------------------
case_labels=()
case_vit_mean=();  case_vit_std=();  case_vit_min=();  case_vit_max=()
case_pe_mean=();   case_pe_std=()
case_tot_mean=();  case_tot_std=();  case_tot_min=();  case_tot_max=()
case_first_text=()
case_cache_mem=()

stats() {
    printf '%s\n' "$@" | awk '
        { x[NR]=$1; s+=$1; if (NR==1||$1<min) min=$1; if (NR==1||$1>max) max=$1 }
        END {
            n=NR; mean=s/n; ss=0;
            for (i=1; i<=n; i++) ss += (x[i]-mean)*(x[i]-mean);
            std = (n>1) ? sqrt(ss/(n-1)) : 0;
            printf "%.4f %.4f %.4f %.4f\n", mean, std, min, max;
        }'
}

wait_for_server() {
    local url="$1"
    for _ in $(seq 1 60); do
        if curl -sf "$url/health" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}

run_one_case() {
    local label="$1"; shift
    local envs=("$@")

    local raw_log="$RUN_DIR/${label}.raw.log"
    local server_log="$RUN_DIR/${label}.server.log"
    local summary="$RUN_DIR/${label}.summary.txt"
    : > "$raw_log"; : > "$server_log"

    local vit_arr=() pe_arr=() tot_arr=()
    local out_text=""

    echo "===== case: $label  envs: ${envs[*]:-<none>} ====="

    # boot server (stdbuf -oL -eL forces line-buffered stdout/stderr so the
    # "mtmd batch encoded in X ms" log line lands on disk before bench writes
    # the run-end marker; without it the awk range below misses the log)
    local stdbuf_prefix=()
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf_prefix=(stdbuf -oL -eL)
    fi
    env ${envs[@]+"${envs[@]}"} \
        "${stdbuf_prefix[@]}" "$LLAMA_SERVER_BIN" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --host 127.0.0.1 \
            --port "$PORT" \
            -c 16384 \
            --no-warmup \
            -v >"$server_log" 2>&1 &
    local server_pid=$!
    trap 'kill '"$server_pid"' 2>/dev/null || true; wait '"$server_pid"' 2>/dev/null || true' EXIT

    if ! wait_for_server "http://127.0.0.1:$PORT"; then
        echo "[bench_vlm_server] server failed to start; tail of server log:" >&2
        tail -50 "$server_log" >&2
        kill "$server_pid" 2>/dev/null || true
        exit 1
    fi

    for i in $(seq 1 "$N_RUNS"); do
        # one session = N_TURNS sequential requests over the same N_IMAGES.
        # accumulate ViT/prompt_eval/total across the turns to report the
        # full session cost (the metric a product owner cares about).
        local sess_vit=0 sess_pe=0 sess_tot=0

        for t in $(seq 1 "$N_TURNS"); do
            local resp_file="$RUN_DIR/.tmp.${label}.${i}.${t}.resp.json"
            local payload_file="$RUN_DIR/.tmp.${label}.${i}.${t}.payload.json"

            # vary the prompt per (run, turn) so the server prompt-cache cannot
            # reuse a previous request and skip the ViT encode entirely. The
            # ViT cache uses image-content hash, so it does NOT care about the
            # text prefix and can still hit on turns 2..N within a session.
            build_payload "[run=${label}-${i}-turn=${t}] $PROMPT" "$payload_file"

            local log_lines_before
            log_lines_before=$(wc -l < "$server_log" 2>/dev/null || echo 0)

            curl -sS \
                -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
                -H "Content-Type: application/json" \
                --data-binary @"$payload_file" \
                > "$resp_file" || {
                    echo "[bench_vlm_server] $label run $i turn $t: curl failed"
                    tail -20 "$server_log"
                    exit 1
                }

            local pe tot vit
            pe=$(jq -r '.timings.prompt_ms // empty'           "$resp_file")
            tot=$(jq -r '(.timings.prompt_ms + .timings.predicted_ms) // empty' "$resp_file")

            vit=""
            for _ in $(seq 1 20); do
                vit=$(tail -n +"$((log_lines_before + 1))" "$server_log" \
                    | grep -oE 'mtmd batch encoded in [0-9]+ ms' | grep -oE '[0-9]+' \
                    | awk '{s+=$1} END{if (NR>0) print s}' || true)
                [[ -n "$vit" ]] && break
                sleep 0.1
            done
            # turns 2..N may legitimately have ViT=0 when cache hits skip the
            # forward pass entirely; fall back to 0 instead of failing.
            [[ -z "$vit" ]] && vit=0

            if [[ -z "$pe" || -z "$tot" ]]; then
                echo "[bench_vlm_server] failed to parse case=$label run=$i turn=$t (vit='$vit' pe='$pe' tot='$tot')"
                echo "  resp tail: $(tail -c 400 "$resp_file")"
                echo "  server log tail (last 30 new lines):"
                tail -n +"$((log_lines_before + 1))" "$server_log" | tail -30
                exit 1
            fi

            sess_vit=$(awk -v a="$sess_vit" -v b="$vit" 'BEGIN{print a+b}')
            sess_pe=$( awk -v a="$sess_pe"  -v b="$pe"  'BEGIN{print a+b}')
            sess_tot=$(awk -v a="$sess_tot" -v b="$tot" 'BEGIN{print a+b}')

            if [[ -z "$out_text" && "$t" == "1" ]]; then
                out_text=$(jq -r '.choices[0].message.content // empty' "$resp_file" \
                    | sed -E 's/^[[:space:]]+|[[:space:]]+$//' \
                    | tr '\n' ' ' \
                    | cut -c1-140 || true)
            fi

            printf "  run %2d/%d turn %d/%d  ViT=%s ms  prompt_eval=%.1f ms  total=%.1f ms\n" \
                "$i" "$N_RUNS" "$t" "$N_TURNS" "$vit" "$pe" "$tot"

            rm -f "$resp_file" "$payload_file"
        done

        printf "  run %2d/%d  SESSION  ViT=%.1f ms  prompt_eval=%.1f ms  total=%.1f ms\n" \
            "$i" "$N_RUNS" "$sess_vit" "$sess_pe" "$sess_tot"
        vit_arr+=("$sess_vit"); pe_arr+=("$sess_pe"); tot_arr+=("$sess_tot")
    done

    # ---- ViT cache memory occupancy (parsed from server log) ----------
    # opt/vit-cache emits "... mem=X.XX MiB ..." per encode; pick the last
    # value seen as the steady-state cache footprint for this case. For the
    # baseline case (no OPT_VIT_CACHE) the line is absent, so report 0.
    local steady_mem
    steady_mem=$(grep -oE 'mem=[0-9]+\.[0-9]+ MiB' "$server_log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    [[ -z "$steady_mem" ]] && steady_mem=0

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    trap - EXIT

    read -r vit_mean vit_std vit_min vit_max < <(stats "${vit_arr[@]}")
    read -r pe_mean  pe_std  pe_min  pe_max  < <(stats "${pe_arr[@]}")
    read -r tot_mean tot_std tot_min tot_max < <(stats "${tot_arr[@]}")

    {
        echo "case      : $label"
        echo "envs      : ${envs[*]:-<none>}"
        echo "n_runs    : $N_RUNS"
        echo "n_images  : $N_IMAGES"
        echo "n_turns   : $N_TURNS  (per-session totals reported below)"
        printf "ViT_E2E    : mean=%.2f ms  std=%.2f ms  min=%.2f  max=%.2f\n" "$vit_mean" "$vit_std" "$vit_min" "$vit_max"
        printf "prompt_eval: mean=%.2f ms  std=%.2f ms  min=%.2f  max=%.2f\n" "$pe_mean"  "$pe_std"  "$pe_min"  "$pe_max"
        printf "VLM_E2E    : mean=%.2f ms  std=%.2f ms  min=%.2f  max=%.2f\n" "$tot_mean" "$tot_std" "$tot_min" "$tot_max"
        printf "cache_mem  : %.2f MiB  (steady-state ViT embd cache; 0 if disabled)\n" "$steady_mem"
        echo "raw ViT (ms)    : ${vit_arr[*]}"
        echo "raw prompt (ms) : ${pe_arr[*]}"
        echo "raw total (ms)  : ${tot_arr[*]}"
        echo "first output    : $out_text"
    } | tee "$summary"
    echo

    case_labels+=("$label")
    case_vit_mean+=("$vit_mean"); case_vit_std+=("$vit_std"); case_vit_min+=("$vit_min"); case_vit_max+=("$vit_max")
    case_pe_mean+=("$pe_mean");   case_pe_std+=("$pe_std")
    case_tot_mean+=("$tot_mean"); case_tot_std+=("$tot_std"); case_tot_min+=("$tot_min"); case_tot_max+=("$tot_max")
    case_first_text+=("$out_text")
    case_cache_mem+=("$steady_mem")
}

for line in "${CASES[@]}"; do
    read -r label rest <<< "$line"
    # shellcheck disable=SC2086
    run_one_case "$label" $rest
done

# ----- comparison ------------------------------------------------------
report="$RUN_DIR/comparison.txt"
{
    echo "===================================================================="
    echo " VLM server multi-image bench  env=$ENV_TAG  N_RUNS=$N_RUNS  N_IMAGES=$N_IMAGES  N_TURNS=$N_TURNS  date=$DATE_TAG"
    echo "===================================================================="
    echo "model     : $MODEL"
    echo "image     : $IMAGE  (replicated $N_IMAGES times per request, $N_TURNS turns per session)"
    echo "prompt    : $PROMPT"
    echo "n_predict : $N_PREDICT"
    echo

    base_vit="${case_vit_mean[0]}"
    base_pe="${case_pe_mean[0]}"
    base_tot="${case_tot_mean[0]}"
    base_mem="${case_cache_mem[0]}"

    printf "%-20s | %-22s | %-22s | %-22s | %-12s\n" "case" "ViT_E2E (ms)" "prompt_eval (ms)" "VLM_E2E (ms)" "cache (MiB)"
    printf "%-20s-+-%-22s-+-%-22s-+-%-22s-+-%-12s\n" "--------------------" "----------------------" "----------------------" "----------------------" "------------"
    for i in "${!case_labels[@]}"; do
        local_label="${case_labels[$i]}"
        v="${case_vit_mean[$i]}";  vs="${case_vit_std[$i]}"
        p="${case_pe_mean[$i]}";   ps="${case_pe_std[$i]}"
        t="${case_tot_mean[$i]}";  ts="${case_tot_std[$i]}"
        m="${case_cache_mem[$i]}"

        v_delta=$(awk -v c=$v -v b=$base_vit 'BEGIN{print (b>0)?100*(c-b)/b:0}')
        p_delta=$(awk -v c=$p -v b=$base_pe  'BEGIN{print (b>0)?100*(c-b)/b:0}')
        t_delta=$(awk -v c=$t -v b=$base_tot 'BEGIN{print (b>0)?100*(c-b)/b:0}')
        m_delta=$(awk -v c=$m -v b=$base_mem 'BEGIN{print c-b}')

        if [[ "$i" == "0" ]]; then
            printf "%-20s | %8.1f +- %-7.1f      | %8.1f +- %-7.1f      | %8.1f +- %-7.1f      | %6.2f       \n" \
                "$local_label" "$v" "$vs" "$p" "$ps" "$t" "$ts" "$m"
        else
            printf "%-20s | %8.1f +- %-7.1f %+5.1f%% | %8.1f +- %-7.1f %+5.1f%% | %8.1f +- %-7.1f %+5.1f%% | %6.2f %+5.2f\n" \
                "$local_label" "$v" "$vs" "$v_delta" "$p" "$ps" "$p_delta" "$t" "$ts" "$t_delta" "$m" "$m_delta"
        fi
    done
    echo
    echo "Greedy outputs (should be identical across cases for correctness):"
    for i in "${!case_labels[@]}"; do
        printf "  [%s] %s\n" "${case_labels[$i]}" "${case_first_text[$i]}"
    done
    echo
    echo "raw logs / per-case summaries: $RUN_DIR/"
} | tee "$report"

echo
echo "[bench_vlm_server] done. report -> $report"
