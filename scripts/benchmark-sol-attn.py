#!/usr/bin/env python3
"""Compare SM121 Sol-Attn with the cuDNN backend used by the H3 rig."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sol-source",
        type=Path,
        required=True,
        help="Checkout containing the sol_kernel Python package",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed_ms(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = (actual.float() - expected.float()).square().sum().sqrt()
    denominator = expected.float().square().sum().sqrt().clamp_min(1e-12)
    return float((delta / denominator).item())


def main() -> int:
    args = parse_args()
    package = args.sol_source.expanduser().resolve() / "sol_kernel"
    if not package.is_dir():
        raise SystemExit(f"sol_kernel not found below {args.sol_source}")
    sys.path.insert(0, str(package.parent))
    from sol_kernel import sol_attn

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise SystemExit(f"This evaluation targets DGX Spark SM121, got SM{capability[0]}{capability[1]}")
    if args.head_dim != 128:
        raise SystemExit("Sol-Attn requires --head-dim 128")
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    scale = args.head_dim**-0.5

    def cudnn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            output = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        return output.permute(0, 2, 1, 3)

    rows = []
    with torch.inference_mode():
        for tokens in args.tokens:
            if tokens <= 0:
                raise SystemExit("--tokens values must be positive")
            shape = (1, tokens, args.heads, args.head_dim)
            q = torch.randn(shape, device=device, dtype=torch.bfloat16)
            k = torch.randn(shape, device=device, dtype=torch.bfloat16)
            v = torch.randn(shape, device=device, dtype=torch.bfloat16)

            functions = {
                "cudnn": lambda: cudnn(q, k, v),
                "sol_bf16": lambda: sol_attn(q, k, v, scale=scale, tau=1.0),
                "sol_int8_qk": lambda: sol_attn(
                    q,
                    k,
                    v,
                    scale=scale,
                    tau=1.0,
                    int8_qk=True,
                ),
            }
            timings = {
                name: timed_ms(function, warmup=args.warmup, iterations=args.iterations)
                for name, function in functions.items()
            }

            expected = cudnn(q, k, v)
            sol_exact = sol_attn(q, k, v, scale=scale, tau=-100.0)
            sol_sparse = sol_attn(q, k, v, scale=scale, tau=1.0)
            sol_int8 = sol_attn(q, k, v, scale=scale, tau=1.0, int8_qk=True)
            if not all(torch.isfinite(value).all() for value in (expected, sol_exact, sol_sparse, sol_int8)):
                raise RuntimeError(f"non-finite output at {tokens} tokens")

            row = {
                "tokens": tokens,
                "cudnn_ms": timings["cudnn"],
                "sol_bf16_ms": timings["sol_bf16"],
                "sol_int8_qk_ms": timings["sol_int8_qk"],
                "sol_bf16_speedup_vs_cudnn": timings["cudnn"] / timings["sol_bf16"],
                "sol_int8_qk_speedup_vs_cudnn": timings["cudnn"] / timings["sol_int8_qk"],
                "sol_exact_relative_l2_vs_cudnn": relative_l2(sol_exact, expected),
                "sol_bf16_relative_l2_vs_cudnn": relative_l2(sol_sparse, expected),
                "sol_int8_qk_relative_l2_vs_cudnn": relative_l2(sol_int8, expected),
            }
            if not all(math.isfinite(value) for key, value in row.items() if key != "tokens"):
                raise RuntimeError(f"non-finite measurement at {tokens} tokens")
            rows.append(row)
            del q, k, v, expected, sol_exact, sol_sparse, sol_int8
            torch.cuda.empty_cache()

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": f"SM{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
