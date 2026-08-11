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

MODEL_ROOT="${MINIMAX_H3_MODEL_ROOT:-}"
[[ -n "$MODEL_ROOT" ]] || fail "MINIMAX_H3_MODEL_ROOT is unset"
[[ "$MODEL_ROOT" = /* ]] || fail "MINIMAX_H3_MODEL_ROOT must be an absolute path"
RIG_MODE="${H3_RIG_MODE:-fl2va}"
case "$RIG_MODE" in
  fl2va)
    PARTITION="${H3_FL2VA_PARTITION:-FL2VA-TutuCurve-Step100/model}"
    case "$PARTITION" in
      FL2VA|FL2VA-TutuCurve-Step100/model) ;;
      *) fail "H3_FL2VA_PARTITION must be FL2VA or FL2VA-TutuCurve-Step100/model" ;;
    esac
    ;;
  ref2va) PARTITION="Ref2VA" ;;
  *) fail "H3_RIG_MODE must be fl2va or ref2va" ;;
esac
MODEL_DIR="$MODEL_ROOT/$PARTITION"
[[ -f "$MODEL_DIR/model_index.json" ]] || fail "model_index.json is missing from $MODEL_DIR"
[[ -d "$MODEL_DIR/transformer" ]] || fail "transformer directory is missing from $MODEL_DIR"
TRANSFORMER_INDEX="$MODEL_DIR/transformer/model.safetensors.index.json"
TRANSFORMER_CONFIG="$MODEL_DIR/transformer/config.json"
[[ -f "$TRANSFORMER_INDEX" ]] || fail "transformer weight index is missing from $MODEL_DIR/transformer"
[[ -f "$TRANSFORMER_CONFIG" ]] || fail "transformer config is missing from $MODEL_DIR/transformer"
python3 - "$TRANSFORMER_INDEX" "$TRANSFORMER_CONFIG" "$MODEL_DIR/transformer" <<'PY' ||
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
weights_dir = Path(sys.argv[3])
weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
files = sorted(set(weight_map.values()))
missing = [name for name in files if not (weights_dir / name).is_file()]
if not files or missing:
    detail = ", ".join(missing[:5]) if missing else "weight_map is empty"
    raise SystemExit(f"incomplete transformer checkpoint: {detail}")
config = json.loads(config_path.read_text())
grid = config.get("adaln_curve_grid")
dimensions = config.get("time_embed_dim")
if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2:
    raise SystemExit("canonical partition is not AdaLN curve-compressed")
if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
    raise SystemExit("curve-compressed partition has an invalid time_embed_dim")
PY
  fail "production transformer checkpoint is incomplete or not curve-compressed in $MODEL_DIR/transformer"

LORA_ROOT="${MINIMAX_H3_LORA_ROOT:-}"
[[ -n "$LORA_ROOT" && "$LORA_ROOT" = /* ]] || fail "MINIMAX_H3_LORA_ROOT must be an absolute path"
[[ -d "$LORA_ROOT" ]] || fail "MINIMAX_H3_LORA_ROOT does not exist: $LORA_ROOT"

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

RUNNING="$(docker inspect minimax-h3-rig --format '{{.State.Running}}' 2>/dev/null || true)"
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
printf '  rig mode: %s\n' "$RIG_MODE"
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
