# Measured one-Spark results

These results describe MiniMax H3 FL2VA and the specifically identified
Ref2VA experiments on one NVIDIA DGX Spark. They are not vendor benchmarks,
an upstream support guarantee, or a cross-machine claim.
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
| Optional FP8 linear backend | Deterministic Humming, eager/no-cache only |

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

## Ref2VA reference-image pixel budget

On 2026-08-08, the Ref2VA image built from this repository was tested with one
matched image-only request after startup warmup. The source was a deliberately
difficult 1672x941 composite character image containing two small faces,
distinct hair colors, elongated joined anatomy, lace clothing, microphones,
a mixing surface, and a detailed studio background. The prompt, 768x448 output,
20 steps, four-second duration, flow shifts, and seed 42 remained fixed.

| Reference policy | Encoded reference | Approx. reference latent positions | Logged Qwen presentation tokens | Client | Engine | Mean step |
|---|---:|---:|---:|---:|---:|---:|
| Legacy 2048 short edge | 3648x2048 | 29,184 | 7,190 | 636.913 s | 632.686 s | 31.634 s |
| 200% output pixel area | 1120x608 | 2,660 | 830 | 268.877 s | 266.530 s | 13.327 s |
| 100% output pixel area | 768x448 | 1,344 | 486 | **251.181 s** | **248.878 s** | **12.444 s** |

All three requests returned HTTP 200, passed a complete FFmpeg decode, and
contained 107 H.264 frames at 24 fps plus AAC-LC stereo audio at 32 kHz. The
100% policy reduced client time by 60.56% (2.54x generation rate) versus the
legacy policy. The 200% policy reduced it by 57.78% (2.37x), while costing only
17.696 seconds more than 100%.

Start, midpoint, end, and one-frame-per-second contact sheets were inspected.
Both budgeted outputs retained the two faces, impossible neck geometry, hair
colors, clothing, microphones, console, lighting, and background without an
obvious systematic fidelity loss relative to legacy. Gesture and eye choices
differed, as expected when a conditioning representation changes. This is one
stress fixture rather than a universal quality proof. The product integration
therefore selects 100% for ordinary full-frame references and offers 200% for
many small subjects or unusually fine detail; the low-level patch preserves
legacy behavior when the request field is omitted.

The reproducible request harness is
[`scripts/benchmark-ref2va-pixel-budget.sh`](../scripts/benchmark-ref2va-pixel-budget.sh).
Generated media and the private source fixture are not distributed.

## SM121 Sol-Attn feasibility check

Sol-Attn was evaluated separately against the rig's actual cuDNN full-compute
backend, rather than treating comparisons against SageAttention as transferable.
The source checkout was `Saganaki22/ComfyUI-sol-attn` commit
`90467f1c633ce53af7d77e4a1cc243b5001d89b0`, with H3-shaped BF16 tensors
(`B=1`, `H=56`, `D=128`), seed 42, three warmups, and the median of ten timed
iterations on PyTorch 2.11.0+cu130.

| Tokens | cuDNN | Sol BF16 | BF16 speedup | Sol int8 Q/K | int8 speedup | Sol exact rel. L2 | Sparse BF16 rel. L2 | Sparse int8 rel. L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 20.538 ms | 10.000 ms | 2.05x | 9.198 ms | 2.23x | 0.000168 | 0.747955 | 0.747994 |
| 16,384 | 82.115 ms | 35.113 ms | 2.34x | 30.266 ms | 2.71x | 0.000199 | 0.747746 | 0.747785 |
| 32,768 | 330.062 ms | 129.007 ms | 2.56x | 110.033 ms | 3.00x | 0.000231 | 0.754727 | 0.754765 |

The all-exact Sol setting (`tau=-100`) closely matched cuDNN, validating the
SM121 kernel path and the comparison harness. The default sparse setting
(`tau=1`) was materially approximate on the random fixture. Relative L2 on
random activations is not a video-quality score, but a difference this large
does not meet the project's behavior-preserving adoption bar. No Sol-Attn code
was vendored and it is not a production default. A future visual branch may
evaluate scheduled sparsity, dense early/late steps, and exact conditioning
blocks before any product integration.

The reproducible microbenchmark is
[`scripts/benchmark-sol-attn.py`](../scripts/benchmark-sol-attn.py); it requires
an explicit external Sol-Attn checkout.

## Cumulative optimization A/B

On 2026-08-05, the same request above was repeated around the four cumulative
H3 inference optimizations described in [PATCH.md](PATCH.md). The baseline was
captured from the already-running compatibility image. The optimized image was
then cold-started and rendered twice without changing the request.

| Measurement | Baseline | Optimized first | Optimized repeat |
|---|---:|---:|---:|
| HTTP wall time | 168.169 s | 158.221 s | 151.578 s |
| Server generation stage | 157.536 s | 155.366 s | 148.049 s |
| Mean denoising step | 7.877 s | 7.768 s | 7.402 s |
| MP4 response encoding | 10.602 s | 2.723 s | 3.493 s |
| Full FFmpeg decode | Passed | Passed | Passed |
| Video frames | 56 | 56 | 56 |
| Audio | AAC-LC stereo, 32 kHz | Same | Same |

Relative to the captured baseline, optimized server generation improved by
1.38% on the first request and 6.02% on the immediate repeat. HTTP wall time
improved by 5.92% and 9.87%, respectively, but most of the first wall-time
difference came from variable MP4 encoding and must not be attributed to the
model changes.

The two optimized renders decoded to byte-identical raw video frames. Compared
with the baseline, the optimized trajectory was visually coherent and
comparably detailed but not pixel-identical (video SSIM 0.745). Diffusion can
amplify small floating-point changes, so this is an output-integrity check, not
a formal quality equivalence result. The earlier acceptance timing on this
same machine also demonstrates material system variance; use repeated local
measurements before making capacity claims.

## Historical eager-only Humming FP8 GEMM A/B

The four dominant linear shapes were benchmarked with the same online-FP8
contract used by H3: dynamic per-token E4M3 activations, static per-tensor E4M3
weights, BF16 output, and FP32 accumulation. Humming's batch-invariant
scheduler completed the four-linears set in approximately 74.2 ms versus 81.6
ms for CUTLASS in the five-launch isolated test. The large QKV projection was
bit-repeatable with the batch-invariant scheduler and changed by as much as
0.0625 BF16 units between launches with Humming's default Stream-K schedule.

The full model was then cold-started in the older cuDNN/eager/no-cache profile
with corrected Humming activation-schema and weight-layout handling. It
reported 89.1669 GiB and 479.475 seconds for model loading. The fixed-seed
acceptance request was rendered twice.

| Measurement | CUTLASS + cuDNN repeat | Humming first | Humming repeat |
|---|---:|---:|---:|
| HTTP wall time | 126.839 s | 147.769 s | 121.472 s |
| Server generation stage | 125.241 s | 144.091 s | 117.494 s |
| Mean denoising step | 6.262 s | 7.205 s | 5.875 s |
| MP4 response encoding | 1.574 s | 3.550 s | 3.946 s |
| Full FFmpeg decode | Passed | Passed | Passed |
| Video frames | 56 | 56 | 56 |
| Audio | AAC-LC stereo, 32 kHz | Same | Same |

The first Humming request includes JIT work. On the immediate repeat, server
generation and mean denoise-step latency improved by 6.19% relative to the
profile-matched eager CUTLASS result; HTTP wall time improved by 4.23% because
MP4 encoding varied independently.

The two Humming renders had byte-identical decoded video frames. Their decoded
audio differed only at approximately 164 dB PSNR, consistent with negligible
AAC encoder-level variation. CUTLASS and Humming follow different valid FP32
accumulation orders, so their diffusion trajectories were not pixel-identical
(video SSIM 0.706). Contact-sheet inspection found both outputs coherent,
detailed, and faithful to the macro PCB soldering prompt; this is an integrity
check, not a formal perceptual-quality equivalence study.

This experiment predates the current regional-compile/CUTLASS reconfirmation
and is not directly comparable with its 111.373-second warmed client mean.
Humming is therefore not selected by default. It remains a deterministic
eager/no-cache alternative for operators who want to reproduce the isolated
comparison; the launcher rejects Humming with compile or Cache-DiT.

## Historical eager-only kernel-cache and readiness validation

On 2026-08-06, the older Humming/eager/no-cache profile was started once with
an empty dedicated kernel cache and recreated once with the populated cache.
Both starts remained unhealthy until a two-step text-only warmup at the
production 768x448x56 shape completed.

| Measurement | Empty cache | Populated cache |
|---|---:|---:|
| Model loading | 494.114 s | 479.222 s |
| Production-shape warmup | 43.624 s | 37.871 s |
| Container start to healthy | 9m59s | 9m29s |
| Cache after readiness | 23.4 MB, 682 files | 22,821,931 bytes, 690 files |

The populated-cache warmup was 5.753 seconds (13.2%) faster. The cache held
Humming, Triton, and TorchInductor artifacts and remained exactly 22,821,931
bytes / 690 files across the second container recreation, demonstrating reuse
without new compilation. Checkpoint loading still dominated total startup and
is not expected to benefit from these kernel caches.

After the empty-cache start, the full 20-step seed-42 request was rendered
twice by `make acceptance`:

| Measurement | First | Repeat |
|---|---:|---:|
| HTTP wall time | 118.691 s | 118.494 s |
| Full FFmpeg decode | Passed | Passed |
| Video | 56 frames, H.264, 768x448, 24 fps | Same |
| Audio | AAC-LC stereo, 32 kHz | Same |

Every decoded video frame was byte-identical. Both decoded audio channels
compared at infinite PSNR, and no kernel-fallback message appeared during the
acceptance window.

The first full request immediately after the populated-cache restart completed
in 120.794 seconds with HTTP 200 and a successful full media decode. It stayed
within the established 118--121 second eager/Humming range, so the first user
request no longer carried that profile's earlier Humming JIT penalty. These
startup figures are not performance evidence for the newer default
regional-compile/CUTLASS profile.

## Current default review-candidate revalidation

On 2026-08-06, the cumulative branch was rebased onto repository main at
`7bd148f` and rebuilt with its documented defaults: cuDNN attention, regional
compile, no cache, CUTLASS FP8, and production-shape startup warmup. This is a
direct revalidation of the review candidate, separate from the historical
eager/Humming measurements above.

Model loading used 89.1659 GiB and took 482.705 seconds. The two-step
768x448x56 warmup then completed in 33.738 seconds, and application readiness
arrived about 9 minutes 2 seconds after container start. The persistent cache
volume was already populated from earlier validation, so this is a
populated-cache observation rather than an empty-cache benchmark.

`make acceptance` then rendered the fixed 20-step seed-42 request twice:

| Measurement | First | Repeat |
|---|---:|---:|
| HTTP wall time | 115.204 s | 113.295 s |
| Full FFmpeg decode | Passed | Passed |
| Video | 56 frames, H.264, 768x448, 24 fps | Same |
| Audio | AAC-LC stereo, 32 kHz | Same |

The 114.250-second client mean is consistent with the prior 111.373-second
full-compute reconfirmation under ordinary run variance. Every decoded video
frame was byte-identical, both decoded audio channels compared at infinite
PSNR, and no cuDNN or linear-kernel fallback was logged during the acceptance
window. The container remained healthy with zero OOM kills and zero restarts.

## Patch tests

Eighteen focused tests passed inside the exact built image, covering:

1. grouped QKV layout normalization;
2. native parameter-loader contract preservation;
3. fused fc1 layout validation;
4. native CUDA FP8 activation-quantizer binding;
5. BF16 AdaLN activation input after FP8 weight conversion;
6. packed attention-mask construction and propagation;
7. static-conditioning branch wiring and raw-input fallback;
8. pipeline-level conditioning preparation;
9. fused SwiGLU selection;
10. direct modality embedding assignment;
11. guarded GB10 bandwidth discovery and unknown-device failure;
12. canonical Humming online-FP8 weight restoration and validation;
13. preservation of dynamic per-token Humming activation scaling;
14. Humming-only batch-invariant scheduler configuration;
15. the MiniMax H3 production warmup shape and text-only conditioning; and
16. shared-engine model-specific warmup settings and completion logging.

The remaining cases cover rejection paths within those categories. Launcher
contract tests additionally verify that unrecognized attention and linear
backends are rejected before model startup.
