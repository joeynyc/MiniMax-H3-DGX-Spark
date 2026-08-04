#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE='vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e'
EXPECTED_BASE_ID='sha256:c3cbf972d026ba07223135b1d6b603edb980aa3123c1ead1dc918f057f21f4e3'

command -v docker >/dev/null || { echo 'docker is required' >&2; exit 1; }
"$ROOT/scripts/check-runtime-pin.sh"

actual_id="$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')"
architecture="$(docker image inspect "$BASE_IMAGE" --format '{{.Architecture}}')"
[[ "$actual_id" == "$EXPECTED_BASE_ID" ]] || {
  printf 'unexpected base-image ID: %s\n' "$actual_id" >&2
  exit 1
}
[[ "$architecture" == arm64 ]] || {
  printf 'unexpected base-image architecture: %s\n' "$architecture" >&2
  exit 1
}

docker run --rm --entrypoint python "$BASE_IMAGE" -c '
import importlib.metadata as metadata
import platform
import torch
import vllm

observed = {
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "vllm": vllm.__version__,
    "vllm_omni": metadata.version("vllm-omni"),
    "transformers": metadata.version("transformers"),
    "diffusers": metadata.version("diffusers"),
}
expected = {
    "architecture": "aarch64",
    "python": "3.12.13",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "cudnn": 91900,
    "vllm": "0.26.0",
    "vllm_omni": "0.1.dev2381+g310b4b477",
    "transformers": "5.14.1",
    "diffusers": "0.38.0",
}
print(observed)
assert observed == expected, (observed, expected)
'

printf 'base_image=%s\nbase_image_id=%s\narchitecture=%s\n' \
  "$BASE_IMAGE" "$actual_id" "$architecture"
echo 'runtime provenance passed'
