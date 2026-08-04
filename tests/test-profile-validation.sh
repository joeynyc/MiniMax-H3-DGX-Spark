#!/usr/bin/env bash
set -euo pipefail

IMAGE="${H3_IMAGE:-minimax-h3-dgx-spark:sm121-fp8}"

expect_refuse() {
  local expected="$1"
  shift
  local output

  if output="$(docker run --rm \
    --entrypoint /usr/local/bin/start-minimax-h3 \
    -e H3_BIND_HOST=127.0.0.1 \
    "$@" \
    "$IMAGE" 2>&1)"; then
    printf 'expected profile validation to refuse: %s\n' "$expected" >&2
    exit 1
  fi
  grep -Fq "$expected" <<<"$output" || {
    printf 'missing expected validation message: %s\n%s\n' "$expected" "$output" >&2
    exit 1
  }
}

expect_refuse 'H3_DIFFUSION_ATTENTION_BACKEND must be' \
  -e H3_DIFFUSION_ATTENTION_BACKEND=unsupported
expect_refuse 'H3_EXECUTION_MODE must be' \
  -e H3_EXECUTION_MODE=unsupported
expect_refuse 'H3_CACHE_BACKEND must be' \
  -e H3_CACHE_BACKEND=unsupported
expect_refuse 'cache configuration requires H3_CACHE_BACKEND=cache_dit' \
  -e H3_CACHE_BACKEND=none -e 'H3_CACHE_CONFIG={"threshold":0.10}'
expect_refuse 'requires an explicit tested H3_CACHE_CONFIG' \
  -e H3_CACHE_BACKEND=cache_dit -e H3_CACHE_CONFIG=
expect_refuse 'H3_CACHE_CONFIG must be a JSON object' \
  -e H3_CACHE_BACKEND=cache_dit -e H3_CACHE_CONFIG=not-json

echo 'container profile validation tests passed'
