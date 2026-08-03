#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
check_pattern() {
  local label="$1"
  local pattern="$2"
  if grep -RInE "$pattern" . \
    --exclude-dir=.git --exclude-dir=.pytest_cache --exclude='*.pyc' \
    --exclude='public-audit.sh'; then
    printf 'public audit failed: %s\n' "$label" >&2
    fail=1
  fi
}

check_pattern 'host-specific absolute path' '/home/|/Users/'
check_pattern 'private-network address' '10\.100\.|192\.168\.'
check_pattern 'email address' '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'
check_pattern 'credential marker' '(api[_-]?key|password|secret|BEGIN [A-Z ]*PRIVATE KEY)[[:space:]]*[:=]'

if find . -path ./.git -prune -o -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.jpg' -o -iname '*.jpeg' \
     -o -iname '*.png' -o -iname '*.safetensors' -o -iname '*.bin' \) \
  -print | grep -q .; then
  echo 'public audit failed: generated media or model-like binary found' >&2
  fail=1
fi

if find . -path ./.git -prune -o -type f -size +5M -print | grep -q .; then
  echo 'public audit failed: file larger than 5 MiB found' >&2
  fail=1
fi

(( fail == 0 )) || exit 1
echo 'public audit passed'
