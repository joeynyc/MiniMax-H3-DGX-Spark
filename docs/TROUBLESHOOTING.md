# Troubleshooting

Start with `make status` and container logs. H3 cold loading takes several
minutes; a live container loading checkpoint shards is not a crash.

| Symptom | Likely cause | Action |
|---|---|---|
| Preflight reports insufficient memory | Another model or GPU workload is active | Inventory processes and stop only workloads you own and intend to replace |
| `weight_loader() takes 2 positional arguments but 3 were given` | The unpatched H3 loader is being used | Confirm the custom image built and the patched module is present |
| `Int8 not supported on SM121` | INT8 path selected | Use the included online-FP8 configuration |
| Triton reports `get_fp8e4nv` or FP8 scalar lowering failure | The native-CUDA quantizer binding is missing or a different image is in use | Confirm the pinned image and patched module; eager mode is a diagnostic fallback |
| FP8 quantization kernel rejects `Float8_e4m3fn` input | AdaLN activation was cast to FP8 before the linear | Confirm the BF16 AdaLN patch and test pass |
| CuTe/FA4 variable-length compile error | FlashAttention-4 selected for the packed H3 diffusion shape | Use verified `CUDNN_ATTN` or baseline `TORCH_SDPA`; do not select FA4 |
| HTTP succeeds but output is JSON or tiny | API returned an error body saved under a media filename | Use the included smoke script, which stages and probes the response |
| `/health` works but generation fails | Health does not exercise the diffusion pipeline | Inspect the response error JSON and worker logs; run a real acceptance request |
| First compiled request is slower than later requests | Regional compilation happens on first use | Treat it as warm-up; compare multiple later runs |
| Quality request returns HTTP 504 at a round timeout | `H3_VIDEO_SYNC_TIMEOUT` is shorter than the one-Spark workload | Keep the verified 7,200-second setting or deliberately raise it within the preflight limit |
| Cache-DiT output differs from no-cache output | Cache-DiT intentionally reuses approximate intermediate state | Compare the same seed with `scripts/compare-quality.sh`; disable cache for full compute |
| Container survives but available memory is very low | Another workload, cache, or profile consumed unified memory | Stop the test, inventory the host, and do not publish that profile as stable |

## Verification order

1. `docker ps` shows the container still running.
2. `/health` returns HTTP 200.
3. `/v1/models` reports `/models/MiniMax-H3/FL2VA`.
4. A real `/v1/videos/sync` request returns HTTP 2xx.
5. `scripts/verify-output.sh` finds both video and audio streams and completes a
   full decode.

Do not declare success from GPU allocation, shard loading, or health alone.
For performance work, also require zero OOM kills/restarts, multiple warmed
runs, continuous memory/swap sampling, and the comparison rules in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
