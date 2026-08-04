#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?usage: $0 PROFILE_LABEL}"
[[ "$PROFILE" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "profile label must use lowercase letters, digits, dot, underscore, or hyphen" >&2
  exit 2
}

WARM_RUNS="${H3_BENCH_WARM_RUNS:-2}"
[[ "$WARM_RUNS" =~ ^[1-9][0-9]*$ ]] || {
  echo "H3_BENCH_WARM_RUNS must be a positive integer" >&2
  exit 2
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="$ROOT/output/benchmarks/$STAMP-$PROFILE"
SAMPLES="$RESULT_DIR/memory-swap.csv"
SUMMARY="$RESULT_DIR/summary.txt"
mkdir -p "$RESULT_DIR"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

[[ "${MINIMAX_H3_LICENSE_ACKNOWLEDGED:-false}" == true ]] || {
  echo "model-license acknowledgment is required" >&2
  exit 1
}

monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT INT TERM

monitor() {
  echo 'utc_epoch,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,container_running,oom_killed,restart_count' >"$SAMPLES"
  while true; do
    read -r mem_total mem_available swap_total swap_free < <(
      awk '
        /MemTotal:/ {mt=$2}
        /MemAvailable:/ {ma=$2}
        /SwapTotal:/ {st=$2}
        /SwapFree:/ {sf=$2}
        END {print mt, ma, st, sf}
      ' /proc/meminfo
    )
    state="$(docker inspect -f '{{.State.Running}},{{.State.OOMKilled}},{{.RestartCount}}' minimax-h3-fl2va 2>/dev/null || echo 'false,false,0')"
    printf '%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$mem_total" "$mem_available" "$swap_total" "$swap_free" "$state" >>"$SAMPLES"
    sleep 1
  done
}

cd "$ROOT"
./scripts/preflight.sh
docker compose down --remove-orphans
monitor &
monitor_pid=$!

start_ns="$(date +%s%N)"
docker compose up -d --no-build --force-recreate

ready=0
for _ in $(seq 1 900); do
  running="$(docker inspect -f '{{.State.Running}}' minimax-h3-fl2va 2>/dev/null || true)"
  oom="$(docker inspect -f '{{.State.OOMKilled}}' minimax-h3-fl2va 2>/dev/null || true)"
  [[ "$oom" != true ]] || {
    docker logs minimax-h3-fl2va >"$RESULT_DIR/container.log" 2>&1 || true
    echo "container was OOM-killed during startup" >&2
    exit 1
  }
  [[ "$running" == true ]] || {
    sleep 2
    continue
  }
  if curl -fsS --max-time 2 http://127.0.0.1:"${H3_API_PORT:-8000}"/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || {
  docker logs minimax-h3-fl2va >"$RESULT_DIR/container.log" 2>&1 || true
  echo "service did not become healthy within 30 minutes" >&2
  exit 1
}
ready_ns="$(date +%s%N)"
cold_start_ms=$(( (ready_ns - start_ns) / 1000000 ))

models_json="$(curl -fsS --max-time 10 http://127.0.0.1:"${H3_API_PORT:-8000}"/v1/models)"
python3 -c 'import json,sys; data=json.loads(sys.argv[1]); ids=[x["id"] for x in data["data"]]; assert ids == ["/models/MiniMax-H3/FL2VA"], ids' "$models_json"
printf '%s\n' "$models_json" >"$RESULT_DIR/models.json"

{
  printf 'profile=%s\n' "$PROFILE"
  printf 'cold_start_ms=%s\n' "$cold_start_ms"
  printf 'attention=%s\n' "${H3_DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"
  printf 'execution=%s\n' "${H3_EXECUTION_MODE:-compile}"
  printf 'cache_backend=%s\n' "${H3_CACHE_BACKEND:-none}"
  printf 'cache_config=%s\n' "${H3_CACHE_CONFIG:-}"
  printf 'health_http=200\nserved_model=/models/MiniMax-H3/FL2VA\n'
  printf 'prompt=%s\n' "${PROMPT:-Macro soldering a PCB under warm bench light, soft room tone.}"
  printf 'width=%s\nheight=%s\nfps=%s\nduration=%s\nsteps=%s\nflow_shift=%s\naudio_flow_shift=%s\nseed=%s\n' \
    "${WIDTH:-768}" "${HEIGHT:-448}" "${FPS:-24}" "${DURATION:-2.0}" \
    "${STEPS:-20}" "${FLOW_SHIFT:-12}" "${AUDIO_FLOW_SHIFT:-3.0}" "${SEED:-42}"
  docker image inspect minimax-h3-dgx-spark:sm121-fp8 --format 'image_id={{.Id}} architecture={{.Architecture}} created={{.Created}}'
  printf 'kernel=%s\n' "$(uname -srmo)"
} >"$SUMMARY"

total_runs=$(( WARM_RUNS + 1 ))
for run in $(seq 0 $(( total_runs - 1 ))); do
  OUT="$RESULT_DIR/run-$run.mp4" LOG="$RESULT_DIR/run-$run.log" ./scripts/smoke-t2va.sh
  docker inspect -f 'run='"$run"' running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}' minimax-h3-fl2va >>"$SUMMARY"
  curl -fsS --max-time 5 http://127.0.0.1:"${H3_API_PORT:-8000}"/health >/dev/null
  curl -fsS --max-time 10 http://127.0.0.1:"${H3_API_PORT:-8000}"/v1/models | python3 -c 'import json,sys; data=json.load(sys.stdin); assert [x["id"] for x in data["data"]] == ["/models/MiniMax-H3/FL2VA"]'
  sha256sum "$RESULT_DIR/run-$run.mp4" >>"$SUMMARY"
done

docker logs --timestamps minimax-h3-fl2va >"$RESULT_DIR/container.log" 2>&1
docker inspect -f 'final_running={{.State.Running}} final_oom={{.State.OOMKilled}} final_restarts={{.RestartCount}} image_id={{.Image}}' minimax-h3-fl2va >>"$SUMMARY"
awk -F, 'NR > 1 {used=$2-$3; swap=$4-$5; if (used>peak_used) peak_used=used; if (swap>peak_swap) peak_swap=swap; if ($3<min_available || min_available==0) min_available=$3} END {printf "peak_used_kib=%d\nmin_available_kib=%d\npeak_swap_used_kib=%d\n", peak_used, min_available, peak_swap}' "$SAMPLES" >>"$SUMMARY"
echo "benchmark completed: $RESULT_DIR"
