#!/usr/bin/env bash
set -euo pipefail

RIG_MODE="${H3_RIG_MODE:-fl2va}"
case "$RIG_MODE" in
  fl2va)
    PARTITION="${H3_FL2VA_PARTITION:-FL2VA-TutuCurve-Step100/model}"
    case "$PARTITION" in
      FL2VA|FL2VA-TutuCurve-Step100/model) ;;
      *)
        printf 'startup security check: H3_FL2VA_PARTITION must be FL2VA or FL2VA-TutuCurve-Step100/model\n' >&2
        exit 1
        ;;
    esac
    ;;
  ref2va) PARTITION="Ref2VA" ;;
  *)
    printf 'startup security check: H3_RIG_MODE must be fl2va or ref2va\n' >&2
    exit 1
    ;;
esac
MODEL_PATH="/models/MiniMax-H3/$PARTITION"
QUANT_CONFIG="$(tr -d '\n' </etc/minimax-h3/fp8-quant.json)"
# shellcheck source=scripts/security-common.sh
source /usr/local/lib/minimax-h3/security-common.sh
BIND_HOST="${H3_BIND_HOST:-127.0.0.1}"
API_PORT="${H3_API_PORT:-8000}"
ALLOW_REMOTE="${H3_ALLOW_REMOTE_API:-false}"
API_KEY="${VLLM_API_KEY:-}"
ATTENTION_BACKEND="${H3_DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"
EXECUTION_MODE="${H3_EXECUTION_MODE:-compile}"
CACHE_BACKEND="${H3_CACHE_BACKEND:-none}"
CACHE_CONFIG="${H3_CACHE_CONFIG:-}"
LINEAR_BACKEND="${H3_LINEAR_BACKEND:-CUTLASS}"
STARTUP_WARMUP="${H3_STARTUP_WARMUP:-true}"

fail() {
  printf 'startup security check: %s\n' "$*" >&2
  exit 1
}

if [[ ! "$API_PORT" =~ ^[0-9]+$ ]] || (( API_PORT < 1 || API_PORT > 65535 )); then
  fail "H3_API_PORT must be an integer from 1 to 65535"
fi
h3_validate_network_security "$BIND_HOST" "$ALLOW_REMOTE" "$API_KEY" ||
  fail "network policy rejected the configuration"

case "$ATTENTION_BACKEND" in
  TORCH_SDPA|CUDNN_ATTN) ;;
  *) fail "H3_DIFFUSION_ATTENTION_BACKEND must be TORCH_SDPA or CUDNN_ATTN" ;;
esac
case "$EXECUTION_MODE" in
  eager|compile) ;;
  *) fail "H3_EXECUTION_MODE must be eager or compile" ;;
esac
case "$CACHE_BACKEND" in
  none|cache_dit) ;;
  *) fail "H3_CACHE_BACKEND must be none or cache_dit" ;;
esac
execution_args=()
if [[ "$EXECUTION_MODE" == eager ]]; then
  execution_args+=(--enforce-eager)
fi
cache_args=()
if [[ "$CACHE_BACKEND" != none ]]; then
  cache_args+=(--cache-backend "$CACHE_BACKEND")
fi
if [[ -n "$CACHE_CONFIG" ]]; then
  python -c 'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value, dict)' "$CACHE_CONFIG" ||
    fail "H3_CACHE_CONFIG must be a JSON object"
  cache_args+=(--cache-config "$CACHE_CONFIG")
fi
if [[ "$CACHE_BACKEND" == none && ( ${#cache_args[@]} -ne 0 ) ]]; then
  fail "cache configuration requires H3_CACHE_BACKEND=cache_dit"
fi
if [[ "$CACHE_BACKEND" == cache_dit && -z "$CACHE_CONFIG" ]]; then
  fail "H3_CACHE_BACKEND=cache_dit requires an explicit tested H3_CACHE_CONFIG"
fi

linear_args=()
case "$LINEAR_BACKEND" in
  CUTLASS)
    linear_args+=(--force-cutlass-fp8)
    ;;
  HUMMING)
    [[ "$EXECUTION_MODE" == eager ]] ||
      fail "H3_LINEAR_BACKEND=HUMMING requires H3_EXECUTION_MODE=eager"
    [[ "$CACHE_BACKEND" == none ]] ||
      fail "H3_LINEAR_BACKEND=HUMMING requires H3_CACHE_BACKEND=none"
    # vLLM-Omni 0.1 rebuilds a worker-local VllmConfig and currently drops the
    # outer --linear-backend setting. Disable only the two compatible kernels
    # that precede Humming in the FP8 registry; unsupported candidates reject
    # this per-token/per-tensor scale combination on their own.
    HUMMING_DISABLED_KERNELS="CutlassFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel"
    if [[ -n "${VLLM_DISABLED_KERNELS:-}" ]]; then
      export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS},${HUMMING_DISABLED_KERNELS}"
    else
      export VLLM_DISABLED_KERNELS="$HUMMING_DISABLED_KERNELS"
    fi
    # Humming's Stream-K schedule is not repeatable for H3's large QKV shape
    # on GB10. This Humming-specific flag selects its deterministic schedule
    # without triggering vLLM's global online-FP8 BF16 dequant fallback.
    export VLLM_HUMMING_BATCH_INVARIANT="${VLLM_HUMMING_BATCH_INVARIANT:-1}"
    ;;
  *)
    fail "H3_LINEAR_BACKEND must be CUTLASS or HUMMING"
    ;;
esac

case "$STARTUP_WARMUP" in
  true)
    # The official H3 input contract requires output durations of at least
    # four seconds. 96 requested frames is exactly four seconds at 24 fps
    # before H3 aligns it to the native 17n+5 temporal boundary.
    export MINIMAX_H3_DUMMY_RUN_NUM_FRAMES=96
    ;;
  false)
    export MINIMAX_H3_DUMMY_RUN_NUM_FRAMES=0
    ;;
  *)
    fail "H3_STARTUP_WARMUP must be true or false"
    ;;
esac

# Tradecraft's production runtime is the curve-compressed checkpoint. Refuse
# to silently load a full AdaLN partition under the canonical FL2VA/Ref2VA
# name; full checkpoints are retained only as offline *-Full references.
TRANSFORMER_CONFIG="$MODEL_PATH/transformer/config.json"
python - "$TRANSFORMER_CONFIG" <<'PY' ||
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    config = json.loads(path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read transformer config {path}: {exc}")
grid = config.get("adaln_curve_grid")
dimensions = config.get("time_embed_dim")
if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2:
    raise SystemExit("canonical H3 partition is not AdaLN curve-compressed")
if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
    raise SystemExit("curve-compressed H3 partition has an invalid time_embed_dim")
PY
  fail "production requires a curve-compressed H3 checkpoint in $MODEL_PATH"

exec vllm serve "$MODEL_PATH" \
  --omni \
  --trust-remote-code \
  --host "$BIND_HOST" \
  --port "$API_PORT" \
  --num-gpus 1 \
  --num-weight-load-threads 2 \
  "${execution_args[@]}" \
  "${cache_args[@]}" \
  --diffusion-attention-backend "$ATTENTION_BACKEND" \
  --diffusion-quantization-config "$QUANT_CONFIG" \
  --stage-init-timeout 1800 \
  --init-timeout 2400 \
  "${linear_args[@]}"
