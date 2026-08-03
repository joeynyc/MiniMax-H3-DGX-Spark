#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 OUTPUT.mp4" >&2
  exit 2
}

MEDIA="$1"
[[ -f "$MEDIA" ]] || {
  echo "missing media file: $MEDIA" >&2
  exit 1
}

command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

VIDEO_STREAMS="$(ffprobe -v error -select_streams v -show_entries stream=index -of csv=p=0 "$MEDIA" | wc -l)"
AUDIO_STREAMS="$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$MEDIA" | wc -l)"
(( VIDEO_STREAMS >= 1 )) || { echo "no video stream found" >&2; exit 1; }
(( AUDIO_STREAMS >= 1 )) || { echo "no audio stream found" >&2; exit 1; }

ffmpeg -v error -i "$MEDIA" -f null -

printf '%s\n' 'full_decode=passed'
ffprobe -v error \
  -show_entries format=format_name,duration,size,bit_rate:stream=index,codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels,duration,bit_rate \
  -of json "$MEDIA"
ffmpeg -hide_banner -i "$MEDIA" -map 0:a:0 -af volumedetect -f null - 2>&1 |
  grep -E 'mean_volume|max_volume' || true
sha256sum "$MEDIA"
