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

ACCEPTANCE_DIR="${ACCEPTANCE_DIR:-$ROOT/output/acceptance}"
AUDIO_PSNR_MIN_DB="${H3_ACCEPTANCE_AUDIO_PSNR_MIN_DB:-120}"
FIRST_VIDEO="$ACCEPTANCE_DIR/seed-42-first.mp4"
REPEAT_VIDEO="$ACCEPTANCE_DIR/seed-42-repeat.mp4"
FIRST_LOG="$ACCEPTANCE_DIR/seed-42-first.log"
REPEAT_LOG="$ACCEPTANCE_DIR/seed-42-repeat.log"
TMP_DIR="$(mktemp -d /tmp/minimax-h3-acceptance.XXXXXX)"
STARTED_AT="$(date --iso-8601=seconds)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$ACCEPTANCE_DIR"

OUT="$FIRST_VIDEO" LOG="$FIRST_LOG" "$ROOT/scripts/smoke-t2va.sh"
OUT="$REPEAT_VIDEO" LOG="$REPEAT_LOG" "$ROOT/scripts/smoke-t2va.sh"

ffmpeg -hide_banner -loglevel error -i "$FIRST_VIDEO" \
  -map 0:v:0 -f framemd5 "$TMP_DIR/first.framemd5"
ffmpeg -hide_banner -loglevel error -i "$REPEAT_VIDEO" \
  -map 0:v:0 -f framemd5 "$TMP_DIR/repeat.framemd5"

if ! cmp -s "$TMP_DIR/first.framemd5" "$TMP_DIR/repeat.framemd5"; then
  echo "fixed-seed decoded video frames differ" >&2
  exit 1
fi

AUDIO_PSNR_OUTPUT="$(
  ffmpeg -hide_banner -i "$FIRST_VIDEO" -i "$REPEAT_VIDEO" \
    -lavfi '[0:a][1:a]apsnr' -f null - 2>&1
)"
mapfile -t AUDIO_PSNR_VALUES < <(
  sed -n 's/.*PSNR ch[0-9][0-9]*: \([^ ]*\) dB.*/\1/p' <<<"$AUDIO_PSNR_OUTPUT"
)
if (( ${#AUDIO_PSNR_VALUES[@]} != 2 )); then
  echo "could not read stereo audio PSNR from FFmpeg" >&2
  exit 1
fi
for value in "${AUDIO_PSNR_VALUES[@]}"; do
  if [[ "$value" != "inf" ]] && ! awk -v value="$value" -v floor="$AUDIO_PSNR_MIN_DB" \
    'BEGIN { exit !(value >= floor) }'; then
    printf 'decoded audio PSNR %s dB is below %s dB\n' "$value" "$AUDIO_PSNR_MIN_DB" >&2
    exit 1
  fi
done

SERVER_LOG="$TMP_DIR/server.log"
docker logs --since "$STARTED_AT" minimax-h3-rig >"$SERVER_LOG" 2>&1
if rg -qi 'No available kernel|falling back|fallback kernel' "$SERVER_LOG"; then
  echo "server logged a kernel fallback during fixed-seed acceptance" >&2
  rg -ni 'No available kernel|falling back|fallback kernel' "$SERVER_LOG" >&2
  exit 1
fi

FIRST_MS="$(sed -n 's/^elapsed_ms=//p' "$FIRST_LOG" | tail -1)"
REPEAT_MS="$(sed -n 's/^elapsed_ms=//p' "$REPEAT_LOG" | tail -1)"
VIDEO_HASH="$(sha256sum "$TMP_DIR/repeat.framemd5" | awk '{print $1}')"

printf '%s\n' \
  'fixed_seed_video_frames=identical' \
  "decoded_video_sha256=$VIDEO_HASH" \
  "audio_psnr_ch0_db=${AUDIO_PSNR_VALUES[0]}" \
  "audio_psnr_ch1_db=${AUDIO_PSNR_VALUES[1]}" \
  "first_elapsed_ms=$FIRST_MS" \
  "repeat_elapsed_ms=$REPEAT_MS" \
  'kernel_fallbacks=none' \
  'acceptance=passed'
