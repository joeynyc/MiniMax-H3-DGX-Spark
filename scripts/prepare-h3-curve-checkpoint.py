#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a vLLM H3 curve checkpoint from official and Comfy weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _is_adaln_projection(name: str) -> bool:
    return ".adaln_proj.linear." in name


def _load_curve_tensors(path: Path, required: set[str]):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"curve checkpoint is missing {len(missing)} required tensors: "
                f"{missing[:5]}"
            )
        for name in sorted(required):
            tensors[name] = checkpoint.get_tensor(name).clone()
    return tensors


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(
        os.path.relpath(source, start=destination.parent),
        target_is_directory=source.is_dir(),
    )


def prepare_curve_checkpoint(
    *,
    base_model: Path,
    curve_checkpoint: Path,
    output_model: Path,
) -> dict[str, int]:
    base_model = base_model.resolve()
    curve_checkpoint = curve_checkpoint.resolve()
    output_model = output_model.resolve()
    transformer = base_model / "transformer"
    index_path = transformer / "model.safetensors.index.json"
    config_path = transformer / "config.json"
    if output_model.exists():
        raise FileExistsError(f"output model already exists: {output_model}")
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"base model does not contain a sharded transformer: {transformer}"
        )
    if not curve_checkpoint.is_file():
        raise FileNotFoundError(f"curve checkpoint does not exist: {curve_checkpoint}")

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    adaln_names = {name for name in weight_map if _is_adaln_projection(name)}
    curve_names = adaln_names | {"adaln_t_table"}
    curve_tensors = _load_curve_tensors(curve_checkpoint, curve_names)
    table = curve_tensors["adaln_t_table"]
    if table.ndim != 2 or table.shape[0] < 2:
        raise ValueError(f"adaln_t_table must be [grid >= 2, dimensions], got {table.shape}")
    for name in adaln_names:
        tensor = curve_tensors[name]
        if name.endswith(".weight") and tensor.shape[1] != table.shape[1]:
            raise ValueError(
                f"{name} curve dimension {tensor.shape[1]} does not match table "
                f"dimension {table.shape[1]}"
            )

    output_model.mkdir(parents=True)
    for child in base_model.iterdir():
        if child.name != "transformer":
            _relative_symlink(child, output_model / child.name)
    output_transformer = output_model / "transformer"
    output_transformer.mkdir()

    config = json.loads(config_path.read_text())
    config["adaln_curve_grid"] = int(table.shape[0])
    config["time_embed_dim"] = int(table.shape[1])
    (output_transformer / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    output_weight_map: dict[str, str] = {}
    total_size = 0
    shard_names = sorted(set(weight_map.values()))
    for shard_number, shard_name in enumerate(shard_names):
        source_shard = transformer / shard_name
        destination_shard = output_transformer / shard_name
        shard_tensors = {}
        with safe_open(source_shard, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            for name in source.keys():
                if name.startswith("time_embedder."):
                    continue
                tensor = curve_tensors[name] if name in adaln_names else source.get_tensor(name)
                shard_tensors[name] = tensor
                output_weight_map[name] = shard_name
                total_size += tensor.numel() * tensor.element_size()
            if shard_number == 0:
                shard_tensors["adaln_t_table"] = table
                output_weight_map["adaln_t_table"] = shard_name
                total_size += table.numel() * table.element_size()
            save_file(shard_tensors, destination_shard, metadata=metadata)

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    (output_transformer / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2) + "\n"
    )
    return {
        "curve_grid": int(table.shape[0]),
        "curve_dimensions": int(table.shape[1]),
        "tensor_count": len(output_weight_map),
        "total_size": total_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--curve-checkpoint", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_curve_checkpoint(
        base_model=args.base_model,
        curve_checkpoint=args.curve_checkpoint,
        output_model=args.output_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
