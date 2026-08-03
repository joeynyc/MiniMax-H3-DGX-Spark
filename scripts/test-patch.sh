#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e"

docker run --rm --entrypoint python \
  -v "$ROOT/patches/minimax_h3_transformer.py:/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py:ro" \
  -v "$ROOT/tests:/tests:ro" \
  "$IMAGE" \
  -m pytest -q -p no:cacheprovider /tests/test_h3_loader_patch.py
