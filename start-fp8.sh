#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/models/MiniMax-H3/FL2VA"
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
  --force-cutlass-fp8 \
  --stage-init-timeout 1800 \
  --init-timeout 2400
