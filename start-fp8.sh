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

fail() {
  printf 'startup security check: %s\n' "$*" >&2
  exit 1
}

if [[ ! "$API_PORT" =~ ^[0-9]+$ ]] || (( API_PORT < 1 || API_PORT > 65535 )); then
  fail "H3_API_PORT must be an integer from 1 to 65535"
fi
h3_validate_network_security "$BIND_HOST" "$ALLOW_REMOTE" "$API_KEY" ||
  fail "network policy rejected the configuration"

exec vllm serve "$MODEL_PATH" \
  --omni \
  --trust-remote-code \
  --host "$BIND_HOST" \
  --port "$API_PORT" \
  --num-gpus 1 \
  --num-weight-load-threads 2 \
  --enforce-eager \
  --diffusion-attention-backend TORCH_SDPA \
  --diffusion-quantization-config "$QUANT_CONFIG" \
  --force-cutlass-fp8 \
  --stage-init-timeout 1800 \
  --init-timeout 2400
