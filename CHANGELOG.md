# Changelog

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
