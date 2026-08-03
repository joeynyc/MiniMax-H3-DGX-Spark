#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

[[ "${MINIMAX_H3_LICENSE_ACKNOWLEDGED:-false}" == "true" ]] || {
  echo "Read MODEL-LICENSE.md and acknowledge the model license before generation." >&2
  exit 1
}

PORT="${H3_API_PORT:-8000}"
BASE_URL="${H3_API_BASE:-http://127.0.0.1:$PORT}"
OUT="${OUT:-$ROOT/output/smoke-t2va.mp4}"
LOG="${LOG:-$ROOT/output/smoke-t2va.log}"
PART="${OUT}.part"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
rm -f "$PART"

START_NS="$(date +%s%N)"
HTTP_CODE="$(curl -sS -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=Macro soldering a PCB under warm bench light, soft room tone." \
  -F "width=768" \
  -F "height=448" \
  -F "num_inference_steps=20" \
  -F "flow_shift=12" \
  -F "seed=42" \
  -F "fps=24" \
  -F 'extra_params={"task":"t2va","duration":2.0,"audio_flow_shift":3.0}' \
  --output "$PART" \
  -w '%{http_code}')"
END_NS="$(date +%s%N)"
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

printf 'HTTP %s\nelapsed_ms=%s\n' "$HTTP_CODE" "$ELAPSED_MS" | tee "$LOG"
if [[ "$HTTP_CODE" != 2* ]]; then
  ERROR_OUT="${OUT%.mp4}.error.json"
  mv -f "$PART" "$ERROR_OUT"
  printf 'generation failed; response saved to %s\n' "$ERROR_OUT" | tee -a "$LOG" >&2
  exit 1
fi

if ! ffprobe -v error -show_entries format=format_name,duration \
  -of default=nw=1 "$PART" >>"$LOG" 2>&1; then
  INVALID_OUT="${OUT%.mp4}.invalid"
  mv -f "$PART" "$INVALID_OUT"
  printf 'generation returned a non-video body; saved to %s\n' "$INVALID_OUT" | tee -a "$LOG" >&2
  exit 1
fi

mv -f "$PART" "$OUT"
"$ROOT/scripts/verify-output.sh" "$OUT" | tee -a "$LOG"
