#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/models/MiniMax-H3/FL2VA"
QUANT_CONFIG="$(tr -d '\n' </etc/minimax-h3/fp8-quant.json)"

exec vllm serve "$MODEL_PATH" \
  --omni \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${H3_API_PORT:-8000}" \
  --num-gpus 1 \
  --num-weight-load-threads 2 \
  --enforce-eager \
  --diffusion-attention-backend TORCH_SDPA \
  --diffusion-quantization-config "$QUANT_CONFIG" \
  --force-cutlass-fp8 \
  --stage-init-timeout 1800 \
  --init-timeout 2400
