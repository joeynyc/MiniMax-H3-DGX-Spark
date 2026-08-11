#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e"
TEST_IMAGE="minimax-h3-dgx-spark:patch-test"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$TEST_IMAGE" \
  "$ROOT"

docker run --rm --entrypoint python \
  -v "$ROOT/tests:/tests:ro" \
  -v "$ROOT/scripts:/scripts:ro" \
  "$TEST_IMAGE" \
  -m pytest -q -p no:cacheprovider \
  /tests/test_h3_loader_patch.py \
  /tests/test_diffusion_lora_mixed_dtype.py \
  /tests/test_h3_curve_checkpoint.py \
  /tests/test_h3_curve_tutu_adapter.py \
  /tests/test_h3_runtime_optimizations.py \
  /tests/test_h3_startup_warmup.py \
  /tests/test_humming_gb10_bandwidth.py \
  /tests/test_humming_online_fp8_layout.py

bash -n "$ROOT/scripts/acceptance-fixed-seed.sh"
printf 'fixed-seed acceptance syntax validation passed\n'
