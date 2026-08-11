#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/security-common.sh"

env_file="$(mktemp)"
trap 'rm -f "$env_file"' EXIT
printf '%s\n' \
  'H3_RIG_MODE=fl2va' \
  'H3_API_PORT=8000' \
  >"$env_file"

export H3_RIG_MODE=ref2va
unset H3_API_PORT
h3_source_env_defaults "$env_file" H3_RIG_MODE H3_API_PORT

[[ "$H3_RIG_MODE" == "ref2va" ]] || {
  printf 'explicit environment override was replaced: %s\n' "$H3_RIG_MODE" >&2
  exit 1
}
[[ "$H3_API_PORT" == "8000" ]] || {
  printf 'unset value did not receive .env default: %s\n' "${H3_API_PORT:-}" >&2
  exit 1
}

echo 'environment-default precedence tests passed'
