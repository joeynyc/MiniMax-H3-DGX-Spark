# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_converter():
    path = Path("/scripts/prepare-h3-curve-checkpoint.py")
    spec = importlib.util.spec_from_file_location("h3_curve_converter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_replaces_full_adaln_and_removes_time_embedder(tmp_path):
    converter = _load_converter()
    base = tmp_path / "Ref2VA"
    transformer = base / "transformer"
    transformer.mkdir(parents=True)
    (base / "text_encoder").mkdir()
    (base / "text_encoder" / "config.json").write_text("{}\n")
    (base / "model_index.json").write_text("{}\n")
    (transformer / "config.json").write_text(
        json.dumps({"time_embed_dim": 3, "timestep_input_dim": 2})
    )

    shard_one = {
        "time_embedder.proj_in.weight": torch.ones(2, 2),
        "blocks.0.adaln_proj.linear.weight": torch.ones(2, 3),
        "blocks.0.adaln_proj.linear.bias": torch.ones(2),
        "blocks.0.attn.out_proj.weight": torch.arange(4).reshape(2, 2),
    }
    shard_two = {
        "final_layer.adaln_proj.linear.weight": torch.ones(1, 3),
        "final_layer.adaln_proj.linear.bias": torch.ones(1),
        "final_layer.video_out.weight": torch.arange(2).reshape(1, 2),
    }
    save_file(shard_one, transformer / "model-00001-of-00002.safetensors")
    save_file(shard_two, transformer / "model-00002-of-00002.safetensors")
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name in shard_one
    }
    weight_map.update(
        {name: "model-00002-of-00002.safetensors" for name in shard_two}
    )
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    curve_path = tmp_path / "curve.safetensors"
    curve_tensors = {
        "adaln_t_table": torch.tensor([[0.0], [0.5], [1.0]]),
        "blocks.0.adaln_proj.linear.weight": torch.tensor([[2.0], [3.0]]),
        "blocks.0.adaln_proj.linear.bias": torch.tensor([4.0, 5.0]),
        "final_layer.adaln_proj.linear.weight": torch.tensor([[6.0]]),
        "final_layer.adaln_proj.linear.bias": torch.tensor([7.0]),
    }
    save_file(curve_tensors, curve_path)
    output = tmp_path / "Ref2VA-Curve"

    result = converter.prepare_curve_checkpoint(
        base_model=base,
        curve_checkpoint=curve_path,
        output_model=output,
    )

    assert result["curve_grid"] == 3
    assert result["curve_dimensions"] == 1
    assert (output / "text_encoder").is_symlink()
    config = json.loads((output / "transformer" / "config.json").read_text())
    assert config["adaln_curve_grid"] == 3
    assert config["time_embed_dim"] == 1
    index = json.loads(
        (output / "transformer" / "model.safetensors.index.json").read_text()
    )
    assert "time_embedder.proj_in.weight" not in index["weight_map"]
    assert "adaln_t_table" in index["weight_map"]
    first_shard = output / "transformer" / "model-00001-of-00002.safetensors"
    with safe_open(first_shard, framework="pt", device="cpu") as converted:
        assert "time_embedder.proj_in.weight" not in converted.keys()
        torch.testing.assert_close(
            converted.get_tensor("blocks.0.adaln_proj.linear.weight"),
            curve_tensors["blocks.0.adaln_proj.linear.weight"],
        )
        torch.testing.assert_close(
            converted.get_tensor("adaln_t_table"),
            curve_tensors["adaln_t_table"],
        )

    with pytest.raises(FileExistsError):
        converter.prepare_curve_checkpoint(
            base_model=base,
            curve_checkpoint=curve_path,
            output_model=output,
        )
