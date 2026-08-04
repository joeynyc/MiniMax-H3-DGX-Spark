#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: $0 REFERENCE.mp4 CANDIDATE.mp4 OUTPUT_DIR" >&2
  exit 2
}

REFERENCE="$1"
CANDIDATE="$2"
OUTPUT_DIR="$3"
[[ -f "$REFERENCE" && -f "$CANDIDATE" ]] || {
  echo "both input videos must exist" >&2
  exit 1
}
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR/frames"

ffmpeg -v info -i "$REFERENCE" -i "$CANDIDATE" \
  -lavfi '[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[test];[ref][test]ssim=stats_file='"$OUTPUT_DIR"'/ssim-frames.log' \
  -f null - 2>"$OUTPUT_DIR/ssim.log"
ffmpeg -v info -i "$REFERENCE" -i "$CANDIDATE" \
  -lavfi '[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[test];[ref][test]psnr=stats_file='"$OUTPUT_DIR"'/psnr-frames.log' \
  -f null - 2>"$OUTPUT_DIR/psnr.log"

duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REFERENCE")"
midpoint="$(awk -v d="$duration" 'BEGIN {printf "%.3f", d / 2}')"
near_end="$(awk -v d="$duration" 'BEGIN {v=d-0.25; if (v<0) v=0; printf "%.3f", v}')"
for spec in "start:0.25" "middle:$midpoint" "end:$near_end"; do
  label="${spec%%:*}"
  timestamp="${spec#*:}"
  ffmpeg -y -v error -ss "$timestamp" -i "$REFERENCE" -frames:v 1 "$OUTPUT_DIR/frames/reference-$label.png"
  ffmpeg -y -v error -ss "$timestamp" -i "$CANDIDATE" -frames:v 1 "$OUTPUT_DIR/frames/candidate-$label.png"
done

ffmpeg_filters="$(ffmpeg -filters 2>/dev/null)"
if grep -q ' apsnr ' <<<"$ffmpeg_filters"; then
  ffmpeg -v info -i "$REFERENCE" -i "$CANDIDATE" \
    -filter_complex '[0:a]aresample=32000:first_pts=0[a0];[1:a]aresample=32000:first_pts=0[a1];[a0][a1]apsnr' \
    -f null - 2>"$OUTPUT_DIR/audio-psnr.log"
else
  ffmpeg -v info -i "$REFERENCE" -i "$CANDIDATE" \
    -filter_complex '[0:a]aresample=32000:first_pts=0[a0];[1:a]aresample=32000:first_pts=0[a1];[a0][a1]amix=inputs=2:weights=1 -1:normalize=0,astats=metadata=1:reset=0' \
    -f null - 2>"$OUTPUT_DIR/audio-difference.log"
fi

grep -E 'SSIM|PSNR' "$OUTPUT_DIR"/*.log || true
echo "quality comparison completed: $OUTPUT_DIR"
