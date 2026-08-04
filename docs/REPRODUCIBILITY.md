# Reproducing the one-Spark measurements

The repository defaults to the fastest validated full-compute profile: cuDNN
attention, regional compile, and no cache. Re-run the sweep after changing the
image digest, patch, model revision, driver, kernel, or runtime settings.

The Compose default sets `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` to 7,200 seconds via
`H3_VIDEO_SYNC_TIMEOUT`. One-Spark 50-step quality requests can exceed the
pinned image's upstream 600-second default. This is a request timeout, not a
performance target; lowering it can cancel healthy long-running generation.

## Runtime identity first

```bash
make provenance
make preflight
make build
```

The provenance check requires the exact digest-pinned ARM64 image and verifies
Python, PyTorch, CUDA, vLLM, and vLLM-Omni versions inside it. The companion
image ID recorded in [RESULTS.md](RESULTS.md) is the measured build identity;
source changes legitimately produce a new companion ID.

## Fixed workload and acceptance gates

`scripts/smoke-t2va.sh` defaults to the published smoke workload: 768x448,
24 fps, two seconds requested, 20 steps, flow shift 12, audio flow shift 3,
seed 42, and the fixed soldering prompt. The benchmark harness records those
inputs, cold-start time, image/kernel identity, per-run client time and hashes,
container logs, and one-second memory/swap samples.

Each accepted profile must satisfy all of these gates:

1. the container remains running with zero OOM kills and zero restarts;
2. `/health` returns HTTP 200;
3. `/v1/models` reports exactly `/models/MiniMax-H3/FL2VA`;
4. a real `POST /v1/videos/sync` T2VA request succeeds;
5. the response contains video and audio streams and fully decodes with FFmpeg;
6. at least two post-warm-up requests complete; and
7. memory/swap sampling retains practical headroom.

Run a profile only on an otherwise idle Spark:

```bash
H3_BENCH_WARM_RUNS=2 ./scripts/benchmark-profile.sh profile-label
```

Set the profile in the ignored `.env` file before each cold start. Benchmark
artifacts are written under ignored `output/benchmarks/` and must not be added
to Git.

## Accepted profiles

Baseline:

```dotenv
H3_DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
H3_EXECUTION_MODE=eager
H3_CACHE_BACKEND=none
H3_CACHE_CONFIG=
```

Full-compute default:

```dotenv
H3_DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN
H3_EXECUTION_MODE=compile
H3_CACHE_BACKEND=none
H3_CACHE_CONFIG=
```

Optional balanced Cache-DiT profile:

```dotenv
H3_DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN
H3_EXECUTION_MODE=compile
H3_CACHE_BACKEND=cache_dit
H3_CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"max_cached_steps":-1,"residual_diff_threshold":0.10,"max_continuous_cached_steps":1,"enable_taylorseer":false}'
```

The pinned image accepted all of these options directly. Regional compile was
confirmed on 52 repeated `MiniMaxH3DiTBlock` modules. Cache-DiT reported itself
enabled on the MiniMax H3 pipeline. These findings do not establish support in
another image.

## Cache-DiT quality method

Cache-DiT is approximate. Compare it to a no-cache output made with the same
prompt, seed, resolution, frame rate, duration, steps, shifts, attention
backend, and execution mode:

```bash
./scripts/compare-quality.sh \
  output/reference.mp4 \
  output/candidate.mp4 \
  output/quality-comparison
```

The script calculates decoded frame-aligned SSIM and PSNR, compares resampled
audio, and extracts start/middle/end frame pairs. Inspect those frames and the
complete videos; aggregate metrics alone cannot establish perceptual
equivalence. Do not call cached output lossless.

## Timing rules

- Cold start is container creation to a successful health response.
- First-request client time includes HTTP transfer and response encoding.
- Engine time comes from vLLM-Omni request timing in container logs.
- Regional compile and cache initialization happen on the first request.
- Compare warmed candidates only with warmed baselines.
- Publish individual warm runs and their mean, not the best run alone.

The exact measured evidence is in [RESULTS.md](RESULTS.md).
