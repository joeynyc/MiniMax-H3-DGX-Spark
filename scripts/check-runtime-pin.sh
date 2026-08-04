#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIGEST='sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e'

for file in Dockerfile compose.yaml scripts/test-patch.sh scripts/runtime-provenance.sh; do
  grep -Fq "$DIGEST" "$ROOT/$file" || {
    printf 'verified base-image digest is missing from %s\n' "$file" >&2
    exit 1
  }
done

unexpected="$(
  grep -hEo 'vllm/vllm-omni:minimax-h3@sha256:[0-9a-f]{64}' \
    "$ROOT/Dockerfile" "$ROOT/compose.yaml" "$ROOT/scripts/test-patch.sh" \
    "$ROOT/scripts/runtime-provenance.sh" |
    sort -u |
    grep -Fv "@$DIGEST" || true
)"
[[ -z "$unexpected" ]] || {
  printf 'conflicting MiniMax-H3 image pin found:\n%s\n' "$unexpected" >&2
  exit 1
}

echo 'runtime pin consistency passed'
