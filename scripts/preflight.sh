#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
# shellcheck disable=SC1091
source "$ROOT/scripts/security-common.sh"

fail() {
  printf 'preflight: %s\n' "$*" >&2
  exit 1
}

[[ -f "$ENV_FILE" ]] || fail "copy .env.example to .env and review it first"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

[[ "${MINIMAX_H3_LICENSE_ACKNOWLEDGED:-false}" == "true" ]] ||
  fail "read MODEL-LICENSE.md and set MINIMAX_H3_LICENSE_ACKNOWLEDGED=true only if authorized"

command -v docker >/dev/null || fail "Docker is required"
docker compose version >/dev/null || fail "Docker Compose v2 is required"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "${ALLOW_UNSUPPORTED_ARCH:-false}" != "true" ]]; then
  fail "verified architecture is aarch64; got $ARCH (set ALLOW_UNSUPPORTED_ARCH=true only for deliberate testing)"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ -n "$GPU_NAME" ]] || fail "no NVIDIA GPU was detected"

MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-}"
[[ -n "$MODEL_DIR" ]] || fail "MINIMAX_H3_MODEL_DIR is unset"
[[ "$MODEL_DIR" = /* ]] || fail "MINIMAX_H3_MODEL_DIR must be an absolute path"
[[ -f "$MODEL_DIR/model_index.json" ]] || fail "model_index.json is missing from $MODEL_DIR"
[[ -d "$MODEL_DIR/transformer" ]] || fail "transformer directory is missing from $MODEL_DIR"

CACHE_DIR="${HF_CACHE_DIR:-}"
[[ -n "$CACHE_DIR" && "$CACHE_DIR" = /* ]] || fail "HF_CACHE_DIR must be an absolute path"
[[ -d "$CACHE_DIR" ]] || fail "HF_CACHE_DIR does not exist: $CACHE_DIR"

BIND_HOST="${H3_BIND_HOST:-127.0.0.1}"
h3_validate_network_security \
  "$BIND_HOST" "${H3_ALLOW_REMOTE_API:-false}" "${H3_API_KEY:-}" ||
  fail "network policy rejected the configuration"

if [[ -n "${H3_API_KEY:-}" ]]; then
  ENV_MODE="$(stat -c '%a' "$ENV_FILE")"
  (( (8#$ENV_MODE & 077) == 0 )) ||
    fail ".env contains H3_API_KEY and must not be readable by group or others; run chmod 600 .env"
fi

PORT="${H3_API_PORT:-8000}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  fail "invalid H3_API_PORT: $PORT"
fi

VIDEO_SYNC_TIMEOUT="${H3_VIDEO_SYNC_TIMEOUT:-7200}"
if [[ ! "$VIDEO_SYNC_TIMEOUT" =~ ^[0-9]+$ ]] || (( VIDEO_SYNC_TIMEOUT < 600 || VIDEO_SYNC_TIMEOUT > 14400 )); then
  fail "H3_VIDEO_SYNC_TIMEOUT must be an integer from 600 to 14400 seconds"
fi

ATTENTION_BACKEND="${H3_DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"
EXECUTION_MODE="${H3_EXECUTION_MODE:-compile}"
CACHE_BACKEND="${H3_CACHE_BACKEND:-none}"
CACHE_CONFIG="${H3_CACHE_CONFIG:-}"

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
if [[ -n "$CACHE_CONFIG" ]]; then
  python3 -c 'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value, dict)' "$CACHE_CONFIG" ||
    fail "H3_CACHE_CONFIG must be a JSON object"
fi
if [[ "$CACHE_BACKEND" == none && -n "$CACHE_CONFIG" ]]; then
  fail "cache configuration requires H3_CACHE_BACKEND=cache_dit"
fi
if [[ "$CACHE_BACKEND" == cache_dit && -z "$CACHE_CONFIG" ]]; then
  fail "H3_CACHE_BACKEND=cache_dit requires an explicit tested H3_CACHE_CONFIG"
fi

RUNNING="$(docker inspect minimax-h3-fl2va --format '{{.State.Running}}' 2>/dev/null || true)"
if [[ "$RUNNING" != "true" ]]; then
  AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  REQUIRED_KIB=$((105 * 1024 * 1024))
  (( AVAILABLE_KIB >= REQUIRED_KIB )) ||
    fail "at least 105 GiB available memory is required before cold start"
  if ss -lntH "sport = :$PORT" | grep -q .; then
    fail "port $PORT is already listening"
  fi
fi

printf 'preflight passed\n'
printf '  architecture: %s\n' "$ARCH"
printf '  GPU: %s\n' "$GPU_NAME"
printf '  model: %s\n' "$MODEL_DIR"
printf '  API bind: %s\n' "$BIND_HOST"
printf '  API port: %s\n' "$PORT"
printf '  synchronous request timeout: %s seconds\n' "$VIDEO_SYNC_TIMEOUT"
printf '  attention: %s\n' "$ATTENTION_BACKEND"
printf '  execution: %s\n' "$EXECUTION_MODE"
printf '  cache: %s\n' "$CACHE_BACKEND"
if [[ -n "${H3_API_KEY:-}" ]]; then
  printf '  API authentication: enabled\n'
else
  printf '  API authentication: disabled (loopback only)\n'
fi
