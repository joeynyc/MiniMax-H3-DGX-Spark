# Changelog

## Unreleased

- Added production AdaLN curve-checkpoint support and a converter that combines
  the compact curve tensors with the unchanged official transformer weights.
- Added a local-only Tutu curve-extension converter. It derives the smallest
  bounded joint residual basis, reuses unchanged checkpoint shards, preserves
  source provenance, and emits an unqualified adapter bundle for review.
- Made the measured extended Tutu 8-NFE bundle the default FL2VA partition,
  while retaining the standard curve partition as an allowlisted recovery path.
- Backported vLLM-Omni's mixed-dtype diffusion-LoRA input repair so FP32 H3
  latent inputs can execute BF16 request-scoped adapters without dtype errors.
- Made the measured component-precision profile the sole packaged runtime:
  dynamic transformer FP8, native Qwen3-VL projection NF4, FP16 video VAE,
  and FP32 audio VAE. The extended FL2VA model-load footprint fell from
  77.1392 GiB to 38.5925 GiB on the recorded GB10 production cold start.
- Autocast checkpoint-owned image and video preprocessing at the FP16 video-VAE
  boundary, preventing Ref2VA's internal FP32 tensors from reaching half-weight
  Conv3D layers while preserving FP32 latent normalization afterward.
- Matched preflight configuration precedence to Docker Compose so an explicit
  operator environment override selects the same rig and model that starts,
  while `.env` continues to provide defaults for unset values.
- Made curve-compressed FL2VA/Ref2VA a fail-closed production requirement;
  full checkpoints are offline conversion and comparison sources, not runtime
  profiles.
- Validated canonical Ref2VA startup, image-plus-audio conditioning,
  video-reference conditioning, and a decoded 20-step quality render while
  retaining roughly 28 GiB available memory under inference.
- Added an opt-in Ref2VA reference-image pixel budget that preserves the
  legacy 2,048-pixel short-edge default unless a request explicitly selects
  50-400 percent of the output pixel area.
- Added matched legacy/200%/100% Ref2VA timing and frame-sheet evaluation; the
  100% case was 2.54x the legacy generation rate on the recorded stress fixture.
- Added an SM121 Sol-Attn microbenchmark against the production cuDNN backend;
  sparse Sol was fast but rejected as a default because the controlled output
  difference was too large for the behavior-preserving bar.
- Added a fail-closed two-step 768x448 startup warmup using a valid four-second
  H3 request and a health check that remains closed until it passes.
- Backported the official H3 mixed-input matrix so Ref2VA accepts visual-only,
  video-only, and mixed visual/audio references without forcing audio.
- Persisted Humming, Triton, and TorchInductor kernel caches across container
  recreation.
- Added a fixed-seed acceptance command that compares every decoded video
  frame, stereo audio PSNR, media validity, timings, and kernel-fallback logs.
- Expanded the pinned-image suite to 39 regression tests.

## 1.1.0 - 2026-08-04

- Added independently measured cuDNN-attention and regional-compile controls;
  made the fastest validated no-cache profile the full-compute default.
- Added an optional, explicitly approximate Cache-DiT profile selected at a
  conservative `0.10` threshold after same-seed SSIM, PSNR, audio, and frame
  inspection.
- Added cold-start/warm-run benchmarking, continuous memory/swap and OOM
  capture, quality comparison, and exact runtime-provenance checks.
- Completed matched 1344x768, 50-step full-compute and Cache-DiT requests with
  full decoding, SSIM 0.881367, video PSNR 26.613264 dB, and explicitly
  approximate frame differences; raised the validated sync timeout to 7,200
  seconds after the prior 1,800-second setting cleanly returned HTTP 504.
- Documented all accepted and rejected one-Spark profiles, per-run timings,
  narrow memory headroom, reproducibility rules, and model-license boundaries.
- Added the required FFmpeg prerequisite, with thanks to @riverar for reporting
  its omission in PR #2.

## 1.0.1 - 2026-08-03

- Changed the default API bind from all interfaces to loopback-only.
- Added fail-closed remote-access acknowledgement, bearer-token authentication,
  `.env` permission checks, and authenticated status/smoke requests.
- Expanded the public audit across reachable history to reject private and
  Tailscale/CGNAT addresses, personal paths, credentials, and generated media.
- Documented the firewall, TLS, rate-limit, and `--trust-remote-code` boundaries.

## 1.0.0 - 2026-08-03

- Published the measured single-DGX-Spark FL2VA compatibility recipe.
- Added a digest-pinned vLLM-Omni image with the SM121 loader, FP8 activation,
  AdaLN BF16, and SDPA corrections.
- Added preflight, status, fail-closed smoke, full media verification, exact-image
  regression tests, and public-content auditing.
- Documented measured performance, failed alternatives, upstream attribution,
  limitations, and the separate MiniMax H3 territorial license.
- Excluded model weights and all generated media from the public repository.
