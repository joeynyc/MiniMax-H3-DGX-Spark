# MiniMax H3 on one NVIDIA DGX Spark

[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Platform: DGX Spark](https://img.shields.io/badge/platform-DGX%20Spark-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Architecture: ARM64](https://img.shields.io/badge/architecture-ARM64-informational)](#verified-configuration)
[![GPU: GB10 SM121](https://img.shields.io/badge/GPU-GB10%20SM121-76B900)](#verified-configuration)

A measured compatibility recipe for serving curve-compressed MiniMax H3 FL2VA
or Ref2VA on a single NVIDIA DGX Spark with vLLM-Omni and online FP8.

I built this after the obvious single-Spark paths failed in different ways:
BF16 ran out of practical unified-memory headroom, INT8 reached an unsupported
SM121 kernel, and the first online-FP8 attempt exposed several day-zero loader
and activation bugs. This repository keeps the successful path small,
inspectable, and repeatable.

> [!IMPORTANT]
> MiniMax H3 is **not** licensed under this repository's Apache-2.0 license.
> Its Community License currently excludes the United States, European Union,
> United Kingdom, and Republic of Korea, and also restricts use and display of
> outputs outside its applicable territory. Read [MODEL-LICENSE.md](MODEL-LICENSE.md)
> and obtain any authorization you need from MiniMax before downloading,
> running, or displaying model output. This repository contains no model
> weights and no generated media.

## Verified configuration

| Item | Measured result |
|---|---:|
| Hardware | 1x DGX Spark, GB10, ARM64, SM121, 128 GB-class unified memory |
| Runtime | Pinned `vllm/vllm-omni:minimax-h3` ARM64 image |
| Production checkpoint | MiniMax H3 AdaLN curve FL2VA or Ref2VA; 37.46 GiB transformer files |
| Offline comparison source | Full FL2VA/Ref2VA retained as `*-Full`, never selected by production |
| Quantization | Online dynamic FP8; CUTLASS default, deterministic Humming eager alternative |
| Full-compute default | cuDNN attention + regional compile, no cache |
| Optional balanced profile | cuDNN + regional compile + Cache-DiT at `0.10` |
| Curve Ref2VA model load | 77.1211 GiB; about 28 GiB remained available during a 20-step render |
| Auxiliary-model proof | Ref2VA render and 5.6 GB GPU-resident official `qwen3.5:9b` visual inference both completed; about 22 GiB remained available |
| Historical full-model load | 89.1659 GiB; retained as comparison evidence only |
| Baseline warmed request | 152.911 seconds client mean (sweep, two runs) |
| Full-compute warmed request | 111.373 seconds client mean (final image, two runs) |
| Review-candidate warmed request | 114.250 seconds client mean (two runs) |
| Balanced warmed request | 80.579 seconds client mean (final image, two runs) |
| Matched quality comparison | SSIM 0.881367; video PSNR 26.613264 dB |
| Output shape observed | 768x448, 24 fps, H.264 + AAC stereo |
| Regression tests | 35 passed in the exact pinned image |

These are observations from one machine, not vendor benchmarks or an upstream
support guarantee. Each performance number above uses the same fixed request
and excludes the first compile/cache warm-up. Cache-DiT reuses approximate
intermediate state; it is not lossless. The generated media is deliberately not
published because of the model-license restrictions above.

## Compatibility layer

```mermaid
flowchart LR
    A[Full checkpoint + curve tensors] --> B[Canonical curve checkpoint]
    B --> C[Pinned vLLM-Omni image]
    C --> D[SM121 compatibility + online FP8]
    D --> E[cuDNN attention + regional compile]
    E --> F[OpenAI-compatible video API]
    F --> G[HTTP and FFmpeg verifier]
```

The patch makes four focused corrections:

1. Normalizes grouped checkpoint QKV rows before native parameter loading.
2. Preserves vLLM's native weight-loader signature for online FP8.
3. Binds FP8 activation quantizers to the supported native CUDA operation on
   SM121 instead of the failing compiled wrapper.
4. Keeps AdaLN activations in BF16 after its linear weights become FP8.

Production additionally uses the checkpoint's compact AdaLN curve form. This
removes neither transformer blocks nor general model capacity; it replaces the
large timestep-conditioned AdaLN projections. The launcher fails closed if a
canonical partition is not curve-compressed, so a mode change cannot silently
restore the memory-monopolizing full checkpoint. See
[the measured curve decision](docs/CURVE_CHECKPOINT_EXPERIMENT.md).

FlashAttention-4's CuTe variable-length kernel also failed for this packed H3
shape on SM121. The measured release supports SDPA and cuDNN attention; cuDNN
with regional compile is the full-compute default. See
[docs/PATCH.md](docs/PATCH.md) for the failure chain and why each change exists.

The image additionally includes four upstream-oriented H3 inference
optimizations: request-level packed-mask reuse, fused SwiGLU, static
prompt/RoPE conditioning, and direct modality embedding scatters. These are
kept separate from the SM121 compatibility rationale and are covered by the
same pinned-image regression suite.

The pinned image predates MiniMax H3's official Ref2VA input-matrix update, so
the image backports vLLM-Omni commit `ae7bee0c` (PR #5752). Ref2VA consequently
accepts image-only and video-only references as well as mixed visual/audio
references, subject to the upstream count limits; audio is not mandatory.

The image also repairs three Humming integration gaps: GB10 bandwidth
discovery, online-FP8 weight layout, and dynamic activation-schema
propagation. Its deterministic scheduler passed repeated fixed-seed eager
renders. That historical eager-only experiment is retained as an explicit
alternative, not the default: the newer cuDNN/compile/CUTLASS profile above is
faster in the measured full-compute comparison. Select Humming only with
`H3_EXECUTION_MODE=eager` and `H3_CACHE_BACKEND=none`; exact evidence is in
[docs/RESULTS.md](docs/RESULTS.md).

Startup now performs a two-step warmup at 768x448 with a valid four-second
request before the health check can pass. Linear, Triton, and TorchInductor
artifacts are stored in a dedicated Compose volume, so container recreation
preserves compiled kernels. Set `H3_STARTUP_WARMUP=false` only as an explicit
diagnostic rollback; it restores lazy first-request compilation and weakens
readiness.

## Quick start

Prerequisites:

- NVIDIA DGX Spark or equivalent ARM64 GB10 system
- Docker with Compose and NVIDIA GPU support
- At least 110 GiB of memory available before launch
- An authorized full MiniMax H3 FL2VA and/or Ref2VA source checkpoint
- The matching AdaLN curve checkpoint tensors used by the converter
- Authorization to use MiniMax H3 in your territory
- FFmpeg, including `ffmpeg` and `ffprobe` (required by smoke and verification)

```bash
git clone https://github.com/joeynyc/MiniMax-H3-DGX-Spark.git
cd MiniMax-H3-DGX-Spark
cp .env.example .env
docker compose build
```

### Prepare the canonical checkpoint

The launcher does not serve a full checkpoint. Preserve each authorized full
source under a `*-Full` name, then build the corresponding canonical curve
partition. The converter requires PyTorch and Safetensors; the pinned image
already contains both, so it can be used without changing the host Python
environment. For FL2VA:

```bash
MODEL_ROOT=/absolute/path/to/MiniMax-H3
CURVE_FILE=/absolute/path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors
docker run --rm --entrypoint python \
  -v "$MODEL_ROOT:/models" \
  -v "$CURVE_FILE:/curve/checkpoint.safetensors:ro" \
  -v "$PWD/scripts:/scripts:ro" \
  minimax-h3-dgx-spark:sm121-fp8 \
  /scripts/prepare-h3-curve-checkpoint.py \
  --base-model /models/FL2VA-Full \
  --curve-checkpoint /curve/checkpoint.safetensors \
  --output-model /models/FL2VA
```

For Ref2VA, use `Ref2VA-Full`, `Ref2VA`, and its matching Ref2VA curve file.
The converter refuses to overwrite an existing output. The full source stays
unchanged and supplies the shared encoders and VAEs through relative links, so
keep the `*-Full` directory beside the canonical partition.

The default FL2VA production profile uses the measured Tutu step100 8-NFE
recipe. Build its extended curve bundle from the pinned upstream adapter after
the canonical FL2VA conversion:

```bash
MODEL_ROOT=/absolute/path/to/MiniMax-H3
TUTU_ADAPTER=/absolute/path/to/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/step100
docker run --rm --entrypoint python \
  -v "$MODEL_ROOT:/models" \
  -v "$TUTU_ADAPTER:/tutu:ro" \
  -v "$PWD/scripts:/scripts:ro" \
  minimax-h3-dgx-spark:sm121-fp8 \
  /scripts/prepare-h3-curve-tutu-adapter.py prepare-extended \
  --full-model /models/FL2VA-Full \
  --curve-model /models/FL2VA \
  --source-adapter /tutu \
  --destination /models/FL2VA-TutuCurve-Step100
```

The emitted `model/` partition also supports the ordinary 20-step request when
no adapter is attached, so switching profiles does not reload the checkpoint.

```bash
# Edit .env: model root, rig mode, HF cache path, and license acknowledgment.
# The secure default listens only on 127.0.0.1.

make preflight
make up
make logs
```

Set `H3_RIG_MODE=fl2va` with the default
`H3_FL2VA_PARTITION=FL2VA-TutuCurve-Step100/model`, or set
`H3_RIG_MODE=ref2va` for `MiniMax-H3/Ref2VA`. The allowlisted value `FL2VA`
selects the standard curve partition as an explicit recovery fallback. The
model root is mounted once and the launcher selects the requested task
partition; both selectable forms must be curve-compressed. Switching modes
requires recreating the single-model service. The `*-Full` sources are not
selectable through the launcher.
`MINIMAX_H3_LORA_ROOT` is a read-only mount for request-scoped PEFT adapters and
does not activate any adapter by itself.

Allow up to 15 minutes for a cold start; the current default review candidate
reached application readiness in about nine minutes on the measured machine.
`health: starting` is expected until both model loading and the
production-shape warmup finish. Wait until both status checks pass:

```bash
make status
```

Then, only where your model license permits generation and display:

```bash
make smoke
make verify
# Optional but recommended after image/backend changes: two full fixed-seed
# renders with decoded video, stereo audio, and kernel-fallback comparisons.
make acceptance
```

The default is the fastest validated full-compute profile. The optional
balanced profile was faster in this request, but it is approximate. Enable it
only after reading [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md):

```dotenv
H3_DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN
H3_EXECUTION_MODE=compile
H3_CACHE_BACKEND=cache_dit
H3_CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"max_cached_steps":-1,"residual_diff_threshold":0.10,"max_continuous_cached_steps":1,"enable_taylorseer":false}'
```

The API base defaults to `http://127.0.0.1:8000/v1`; synchronous generation is
served at `POST /v1/videos/sync`. The smoke script stages the response, checks
the HTTP status, rejects JSON/error bodies, and only promotes a file after
`ffprobe` confirms a media container.

Remote access is deliberately opt-in. Read [SECURITY.md](SECURITY.md) before
changing `H3_BIND_HOST`; a non-loopback bind requires both
`H3_ALLOW_REMOTE_API=true` and a strong `H3_API_KEY`. The client scripts send
the configured bearer token automatically. Do not expose this service directly
to the public internet.

## Repository map

| Path | Purpose |
|---|---|
| `Dockerfile` | Builds on the exact verified base-image digest |
| `compose.yaml` | Portable single-GPU service without host-specific paths |
| `patches/` | Apache-2.0 H3 compatibility and inference patches with modification notice |
| `scripts/preflight.sh` | Architecture, memory, model, port, license, and network-safety checks |
| `scripts/security-common.sh` | Shared fail-closed loopback/remote-access policy |
| `scripts/smoke-t2va.sh` | Fail-closed T2VA acceptance request |
| `scripts/acceptance-fixed-seed.sh` | Two-run decoded-frame, audio, and kernel-fallback acceptance gate |
| `scripts/verify-output.sh` | Streams, full decode, audio, and checksum validation |
| `scripts/benchmark-profile.sh` | Cold start, warm requests, identity, OOM, memory, and swap capture |
| `scripts/benchmark-ref2va-pixel-budget.sh` | Matched legacy/200%/100% Ref2VA reference-conditioning comparison |
| `scripts/benchmark-sol-attn.py` | External Sol-Attn checkout versus the rig's cuDNN backend on SM121 |
| `scripts/prepare-h3-curve-checkpoint.py` | Builds the required canonical AdaLN curve checkpoint without modifying its sources |
| `scripts/prepare-h3-curve-tutu-adapter.py` | Derives a local-only Tutu residual adapter and minimal curve-extension overlay for qualification |
| `scripts/compare-quality.sh` | Frame-aligned SSIM/PSNR, audio comparison, and inspection frames |
| `scripts/runtime-provenance.sh` | Exact image ID, architecture, and runtime-version check |
| `tests/` | Focused loader, FP8, AdaLN, packaging, profile, and optimization regressions |
| `docs/RESULTS.md` | Exact measured acceptance evidence |
| `docs/CURVE_CHECKPOINT_EXPERIMENT.md` | AdaLN curve A/B, production decision, and Ref2VA acceptance evidence |
| `docs/REPRODUCIBILITY.md` | Profiles, workload, warm-up, and quality-comparison method |
| `docs/TROUBLESHOOTING.md` | Symptom-to-cause guidance from the bring-up |

## Scope and honesty

- This is a single-Spark H3 path, not a multi-node implementation. Most
  published profiles are FL2VA-specific; the explicitly labeled Ref2VA pixel
  budget experiment is one stress fixture, not a general quality guarantee.
- It uses an AdaLN curve checkpoint with online FP8; the retained full
  checkpoint is an offline comparison source, not the production fallback.
- The pinned image warns that its bundled vLLM and vLLM-Omni versions are not
  aligned. The recorded path passed, but that warning is not suppressed.
- No model weights, Hugging Face credentials, private endpoints, or generated
  outputs belong in this repository.
- The launcher uses `--trust-remote-code`; use only a checkpoint from the
  authoritative model source and read the runtime boundary in `SECURITY.md`.
- Cache-DiT is approximate. Its measured same-seed SSIM/PSNR results are not a
  guarantee for a different prompt, seed, resolution, step count, or runtime.
- Re-run the full acceptance flow before changing the base-image digest,
  attention backend, execution mode, cache settings, linear backend, or patch.

## Upstream and attribution

- [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Upstream H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [H3 support pull request](https://github.com/vllm-project/vllm-omni/pull/5691)
- [Official H3 input-matrix update](https://github.com/vllm-project/vllm-omni/pull/5752)
- Upstream optimization pull requests:
  [mask reuse](https://github.com/vllm-project/vllm-omni/pull/5817),
  [fused SwiGLU](https://github.com/vllm-project/vllm-omni/pull/5818),
  [static conditioning](https://github.com/vllm-project/vllm-omni/pull/5819), and
  [direct embedding scatters](https://github.com/vllm-project/vllm-omni/pull/5820)

Thanks to [@riverar](https://github.com/riverar) for catching the missing
FFmpeg prerequisite in [PR #2](https://github.com/joeynyc/MiniMax-H3-DGX-Spark/pull/2).

The repository code is Apache-2.0. MiniMax H3 weights and outputs remain subject
to MiniMax's separate license. See [NOTICE](NOTICE) for source attribution.
