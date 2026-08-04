# Measured one-Spark results

These results describe MiniMax H3 FL2VA on one NVIDIA DGX Spark. They are not
vendor benchmarks, an upstream support guarantee, or a cross-machine claim.
Generated artifacts, frames, request logs, and model weights are not included.

## Runtime identity

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, SM121 |
| Host architecture | ARM64 (`aarch64`) |
| Kernel | Linux 6.17.0-1029-nvidia |
| GPU count | 1 |
| Base image | `vllm/vllm-omni:minimax-h3` |
| Base-image digest | `sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e` |
| Base-image ID | `sha256:c3cbf972d026ba07223135b1d6b603edb980aa3123c1ead1dc918f057f21f4e3` |
| Sweep companion-image ID | `sha256:b6f01e8d7c26dfbb9430daf812beada546a69f55c4dd63cba11e8a2e0c42ff1c` |
| Final release companion-image ID | `sha256:4ce6f2eac0a2cd68f5e0e41810bd5d69db88e613fad663b175f940fe1ab99525` |
| Python | 3.12.13 |
| PyTorch / CUDA / cuDNN | 2.11.0+cu130 / 13.0 / 9.19.0 (`91900`) |
| NVIDIA driver | 580.173.02 |
| vLLM-Omni | `0.1.dev2381+g310b4b477` |
| vLLM | `0.26.0` |
| Transformers / Diffusers | 5.14.1 / 0.38.0 |
| Model | MiniMax H3 FL2VA, served as `/models/MiniMax-H3/FL2VA` |
| Quantization | Online dynamic FP8; six sensitive layers ignored |
| FP8 linear backend | CUTLASS |

The image emitted a vLLM/vLLM-Omni major/minor mismatch warning. It is recorded
here rather than hidden; this exact pinned combination passed the measured
requests. `make provenance` checks the base ID, architecture, and core runtime
versions.

The optimization sweep used the sweep image above. The final image changed
only launcher defaults, validation, and the synchronous request timeout; the
model patch and pinned base did not change. Full-compute and selected cached
profiles were rerun on the final image before release.

## Fixed smoke workload

```text
task: t2va
prompt: Macro soldering a PCB under warm bench light, soft room tone.
width: 768
height: 448
fps: 24
duration requested: 2.0 seconds
num_inference_steps: 20
flow_shift: 12
audio_flow_shift: 3.0
seed: 42
```

Every candidate started cold, loaded 13/13 shards, reported 89.1659 GiB model
memory, returned HTTP 200, remained healthy, exposed the exact model ID, and
completed a full FFmpeg audio/video decode. Each row has one first request and
two subsequent warm requests. Client time includes HTTP transfer and response
encoding; engine time comes from vLLM-Omni timing. The first regional-compile
request is a compile warm-up and is not compared with warm eager requests.

| Profile | Cold start | First client | First engine | Warm client runs | Warm client mean | Warm engine mean | Warm step mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SDPA + eager, no cache (baseline) | 585.284 s | 155.384 s | 154.377 s | 152.981 / 152.841 s | 152.911 s | 152.160 s | 7.608 s |
| cuDNN + eager, no cache | 569.168 s | 136.587 s | 132.195 s | 130.634 / 132.977 s | 131.806 s | 128.519 s | 6.426 s |
| cuDNN + regional compile, no cache | 581.343 s | 132.466 s | 130.872 s | 115.022 / 113.272 s | 114.147 s | 112.119 s | 5.606 s |
| SDPA + eager + Cache-DiT `0.10` | 579.654 s | 113.560 s | 110.540 s | 109.100 / 109.495 s | 109.298 s | 106.966 s | 5.348 s |
| SDPA + eager + Cache-DiT `0.15` | 569.469 s | 97.724 s | 96.153 s | 94.128 / 93.843 s | 93.986 s | 92.002 s | 4.600 s |
| cuDNN + regional compile + Cache-DiT `0.10` | 579.417 s | 103.019 s | 100.524 s | 83.767 / 83.375 s | **83.571 s** | **80.959 s** | **4.048 s** |

The selected full-compute default reduced warmed client time by 25.35% versus
the measured baseline. The optional balanced profile reduced it by 45.35%
(1.83x generation rate). That larger result includes approximate Cache-DiT
reuse and must not be described as lossless.

### Final release-image full-compute reconfirmation

The final image repeated the no-cache cuDNN/compile profile from cold start.
Cold readiness was 589.346 seconds. The first client/engine times were
129.835/128.135 seconds; warmed client runs were 110.280 and 112.465 seconds
(111.373-second mean), with 108.776 and 109.505 seconds engine time
(109.141-second mean). Warm step latency averaged 5.457 seconds. Both warmed
MP4 hashes matched, all decodes passed, peak used memory was 113.01 GiB,
minimum available memory was 8.68 GiB, peak observed swap was 3.98 GiB, and the
container recorded zero OOM kills or restarts.

### Final release-image cached reconfirmation

The final image repeated the selected cuDNN/compile/Cache-DiT `0.10` profile
from cold start. Cold readiness was 591.663 seconds. The first client/engine
times were 99.702/98.213 seconds; warmed client runs were 80.291 and 80.867
seconds (80.579-second mean), with 78.957 and 79.715 seconds engine time
(79.336-second mean). Warm step latency averaged 3.967 seconds. Both warmed
MP4 hashes matched and all decodes passed. Against the final-image no-cache
reconfirmation, warmed client time was 27.65% lower (1.38x generation rate).
Peak used memory was 113.09 GiB, minimum available memory was 8.60 GiB, peak
observed swap was 3.73 GiB, and the container recorded zero OOM kills or
restarts.

## Selection decisions

- cuDNN/eager passed every gate but was not selected because cuDNN with
  regional compile was faster after equal warm-up.
- Cache-DiT `0.10` with SDPA/eager passed independently before cache was
  combined with cuDNN/compile.
- Cache-DiT `0.15` was faster, but was rejected as the balanced setting because
  same-seed SSIM fell from 0.851013 at `0.10` to 0.813415 and inspected motion
  diverged more.
- The combined `0.10` profile is the fastest stable smoke profile. It remains
  optional because approximate reuse changes the generated video.
- The original 1,800-second synchronous timeout was rejected for the one-Spark
  50-step quality workload after an exact HTTP 504 at 1,800.373 seconds. The
  request aborted cleanly; health, model identity, zero OOM state, and zero
  restarts all remained intact. The release uses 7,200 seconds.

## Memory and service stability

Memory and swap were sampled once per second from cold start through all three
requests. Swap is an observed host total, not an allocation uniquely
attributable to the container.

| Profile | Peak used memory | Minimum available | Peak swap used | OOM kills / restarts |
|---|---:|---:|---:|---:|
| SDPA + eager | 113.92 GiB | 7.77 GiB | 4.04 GiB | 0 / 0 |
| cuDNN + eager | 113.66 GiB | 8.02 GiB | 3.90 GiB | 0 / 0 |
| cuDNN + compile | 112.82 GiB | 8.87 GiB | 10.66 GiB | 0 / 0 |
| Cache-DiT `0.10` | 108.18 GiB | 13.51 GiB | 12.64 GiB | 0 / 0 |
| Cache-DiT `0.15` | 108.59 GiB | 13.10 GiB | 12.57 GiB | 0 / 0 |
| Combined balanced `0.10` | 113.64 GiB | 8.05 GiB | 5.53 GiB | 0 / 0 |

The headroom is narrow. The preflight requires 105 GiB available and the
quick start recommends at least 110 GiB. Do not run this beside another large
model or assume a lower-memory machine is safe.

## Cache-DiT quality comparison

Cache-DiT was tested conservatively at `0.10` before `0.15`. Decoded frames
were aligned by timestamp and compared against the same prompt/seed no-cache
output. Audio was resampled to 32 kHz before PSNR comparison. Start, midpoint,
and end frames were inspected separately.

| Candidate and reference | SSIM | Video PSNR | Audio PSNR (L/R) | Observation |
|---|---:|---:|---:|---|
| SDPA/eager Cache-DiT `0.10` vs SDPA/eager | 0.851013 | 23.490966 dB | 165.803 / 166.064 dB | Coherent subject; visible tool-path and pose differences |
| SDPA/eager Cache-DiT `0.15` vs SDPA/eager | 0.813415 | 23.682629 dB | 166.361 / 166.535 dB | Faster, but lower SSIM and more framing/motion divergence |
| Combined `0.10` vs cuDNN/compile | 0.810367 | 23.456204 dB | 165.349 / 165.497 dB | Coherent subject; chip framing and iron path differ |

No inspected result showed an obvious decode failure or temporal corruption,
but the cached videos are visibly different. `0.15` was rejected as the
balanced default because its SSIM was lower than `0.10`; speed alone was not
treated as the selection criterion.

## Matched 50-step quality workload

The selected final-image profile and its matched no-cache reference were each
run once after their corresponding smoke warm-ups:

```text
task: t2va
prompt: Photorealistic live-action documentary footage, one continuous
  eye-level wide tracking shot of exactly five adult friends walking together
  along a remote tropical beach at golden hour. The group has varied skin
  tones, gender presentation, ages roughly 25 to 50, and body types, all
  wearing ordinary nonsexual casual beach clothing. Keep all five adults
  distinct and fully visible with stable faces, hands, limbs, clothing, and
  consistent identities. Physically coherent surf rolls in and recedes,
  footprints persist in wet sand, reflections follow each person, and wind
  moves hair, loose fabric, and palm leaves in the same direction. Natural
  unposed conversation gestures, no posing, no children, no extra people, no
  cuts, no text, no logos. Realistic ambient ocean and wind audio, no music.
width: 1344
height: 768
fps: 24
duration requested: 4.0 seconds
num_inference_steps: 50
flow_shift: 12
audio_flow_shift: 3.0
seed: 314159
```

| Measurement | Full compute | Cache-DiT `0.10` |
|---|---:|---:|
| Client elapsed | 2,379.370 s | 1,159.493 s |
| Engine elapsed | 2,301.339 s | 1,091.862 s |
| Mean step latency | 46.027 s | 21.837 s |
| Peak used memory | 112.22 GiB | 112.99 GiB |
| Minimum available memory | 9.47 GiB | 8.70 GiB |
| Peak observed swap | 3.71 GiB | 3.87 GiB |
| OOM kills / restarts | 0 / 0 | 0 / 0 |
| HTTP / model / full decode | Passed | Passed |
| Frames / output duration | 107 / 4.482 s | 107 / 4.482 s |

This is a single matched quality pair, so the elapsed values are reported as
observations, not a generalized quality-workload speedup claim. Both outputs
are H.264 1344x768 at 24 fps with AAC-LC stereo audio at 32 kHz.

Frame-aligned comparison measured SSIM 0.881367 and average video PSNR
26.613264 dB. Audio PSNR was 167.862/167.869 dB (left/right). Start, midpoint,
and end inspection showed exactly five coherent adults, stable identities,
surf, footprints, reflections, and shadows in both outputs, with no obvious
decode or temporal artifact. The cached version visibly changed the central
person's shirt and some pose/foot placement, reinforcing that Cache-DiT is
approximate rather than lossless.

## Output acceptance

All smoke outputs contained 56 H.264 frames at constant 24 fps and AAC-LC
stereo audio at 32 kHz. The observed container duration was 2.357 seconds for
a two-second request. A complete decode passed for every output, and repeated
warm runs within each profile produced matching SHA-256 hashes.

The generated artifacts are not distributed in this public repository. See
[the model-license notice](../MODEL-LICENSE.md).

## Patch tests

Five focused tests passed inside the exact base image:

1. grouped QKV layout normalization;
2. native parameter-loader contract preservation;
3. fused fc1 layout validation;
4. native CUDA FP8 activation-quantizer binding; and
5. BF16 AdaLN activation input after FP8 weight conversion.
