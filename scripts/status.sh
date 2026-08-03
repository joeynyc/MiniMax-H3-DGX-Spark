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

printf '%s\n' '== container =='
docker ps -a --filter name=minimax-h3-fl2va \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

printf '\n%s\n' '== memory =='
free -h

printf '\n%s\n' '== health =='
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' "$BASE_URL/health" || true

printf '\n%s\n' '== models =='
curl -fsS "$BASE_URL/v1/models" || true
printf '\n'
