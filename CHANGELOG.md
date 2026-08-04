# Changelog

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
