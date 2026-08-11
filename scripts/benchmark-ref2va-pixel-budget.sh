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

REFERENCE_IMAGE="${REFERENCE_IMAGE:?Set REFERENCE_IMAGE to one local Ref2VA image}"
[[ -f "$REFERENCE_IMAGE" ]] || {
  echo "Reference image not found: $REFERENCE_IMAGE" >&2
  exit 1
}

PORT="${H3_API_PORT:-8000}"
BASE_URL="${H3_API_BASE:-http://127.0.0.1:$PORT}"
OUT_DIR="${OUT_DIR:-$ROOT/output/ref2va-pixel-budget}"
PROMPT="${PROMPT:-subject_definitions: <Subject 1> is the visible subject from <Picture 1>, preserving its recognizable form, colors, materials, proportions, and distinctive details. summary: [reference generation] Create a four-second identity-retention test. retention_analysis: <Subject 1> is fully preserved. detailed_description: [Shot 1] A steady medium close-up of <Subject 1> in the same visual style as <Picture 1>. The subject makes one small natural head movement and one restrained hand gesture while remaining sharply recognizable. Soft neutral studio lighting, stable framing, coherent anatomy, no cuts. overall_soundscape: Quiet room tone and subtle natural movement. non_diegetic_music: None.}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-448}"
STEPS="${STEPS:-20}"
FLOW_SHIFT="${FLOW_SHIFT:-12}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"
DURATION="${DURATION:-4.0}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3.0}"
BUDGETS="${BUDGETS:-legacy 200 100}"
AUTH_ARGS=()
if [[ -n "${H3_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${H3_API_KEY}")
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf 'profile\tpixel_budget_percent\telapsed_ms\thttp_status\toutput\n' > "$SUMMARY"

for budget in $BUDGETS; do
  if [[ "$budget" == "legacy" ]]; then
    label="legacy-2048-short-edge"
    budget_json=""
  elif [[ "$budget" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    label="budget-${budget}-percent"
    budget_json=",\"reference_image_pixel_budget_percent\":$budget"
  else
    echo "Invalid BUDGETS entry: $budget" >&2
    exit 1
  fi

  output="$OUT_DIR/$label.mp4"
  partial="$output.part"
  log="$OUT_DIR/$label.log"
  rm -f "$partial"
  start_ns="$(date +%s%N)"
  http_code="$(curl -sS "${AUTH_ARGS[@]}" -X POST "$BASE_URL/v1/videos/sync" \
    -F "prompt=$PROMPT" \
    -F "width=$WIDTH" \
    -F "height=$HEIGHT" \
    -F "num_inference_steps=$STEPS" \
    -F "flow_shift=$FLOW_SHIFT" \
    -F "seed=$SEED" \
    -F "fps=$FPS" \
    -F "input_reference=@\"$REFERENCE_IMAGE\";filename=reference.png" \
    -F "extra_params={\"task\":\"ref2va\",\"duration\":$DURATION,\"audio_flow_shift\":$AUDIO_FLOW_SHIFT$budget_json}" \
    --output "$partial" \
    -w '%{http_code}')"
  end_ns="$(date +%s%N)"
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

  printf 'HTTP %s\nelapsed_ms=%s\nprofile=%s\npixel_budget_percent=%s\n' \
    "$http_code" "$elapsed_ms" "$label" "$budget" | tee "$log"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$budget" "$elapsed_ms" "$http_code" "$output" >> "$SUMMARY"

  if [[ "$http_code" != 2* ]]; then
    mv -f "$partial" "$OUT_DIR/$label.error.json"
    echo "Generation failed for $label" >&2
    exit 1
  fi
  if ! ffprobe -v error -show_entries format=format_name,duration \
    -of default=nw=1 "$partial" >> "$log" 2>&1; then
    mv -f "$partial" "$OUT_DIR/$label.invalid"
    echo "Generation returned invalid media for $label" >&2
    exit 1
  fi
  mv -f "$partial" "$output"
  "$ROOT/scripts/verify-output.sh" "$output" | tee -a "$log"
done

column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
