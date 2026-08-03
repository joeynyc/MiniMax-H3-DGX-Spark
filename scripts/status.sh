#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

PORT="${H3_API_PORT:-8000}"
BASE_URL="${H3_API_BASE:-http://127.0.0.1:$PORT}"
AUTH_ARGS=()
if [[ -n "${H3_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${H3_API_KEY}")
fi

printf '%s\n' '== container =='
docker ps -a --filter name=minimax-h3-fl2va \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

printf '\n%s\n' '== memory =='
free -h

printf '\n%s\n' '== health =='
curl -fsS "${AUTH_ARGS[@]}" -o /dev/null -w 'HTTP %{http_code}\n' "$BASE_URL/health" || true

printf '\n%s\n' '== models =='
curl -fsS "${AUTH_ARGS[@]}" "$BASE_URL/v1/models" || true
printf '\n'
