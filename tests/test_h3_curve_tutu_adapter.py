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
    path = Path("/scripts/prepare-h3-curve-tutu-adapter.py")
    spec = importlib.util.spec_from_file_location("h3_curve_tutu_converter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model(
    root: Path,
    tensors: dict[str, torch.Tensor],
    config: dict[str, int],
) -> None:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, transformer / shard)
    (transformer / "config.json").write_text(json.dumps(config) + "\n")
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard for name in tensors},
            }
        )
        + "\n"
    )


def test_projection_reconstructs_curve_representable_adaln_delta() -> None:
    converter = _load_converter()
    curve = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-0.5, 1.5]]
    )
    base_features = curve @ torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]]
    )
    adapted_features = curve @ torch.tensor(
        [[1.5, 1.5, -0.5], [0.25, -0.5, 1.5]]
    )
    base_weight = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 0.0]]
    )
    source_a = torch.tensor([[1.0, -0.5, 0.25]])
    source_b = torch.tensor([[1.0], [2.0], [-1.0]])

    curve_a, curve_b, diagnostics = converter.project_adaln_delta(
        curve_coordinates=curve,
        base_features=base_features,
        adapted_features=adapted_features,
        base_weight=base_weight,
        source_lora_a=source_a,
        source_lora_b=source_b,
        source_lora_scale=0.4,
        output_adapter_scale=0.8,
    )

    expected_delta = (
        (adapted_features - base_features) @ base_weight.T
        + 0.4 * (adapted_features @ source_a.T) @ source_b.T
    )
    reconstructed = 0.8 * (curve @ curve_a.T) @ curve_b.T
    torch.testing.assert_close(curve_a, torch.eye(2))
    torch.testing.assert_close(reconstructed, expected_delta, atol=1e-5, rtol=1e-5)
    assert diagnostics["time_feature_relative_error"] < 1e-6
    assert diagnostics["lora_feature_relative_error"] < 1e-6


def test_basis_derivation_adds_the_smallest_shared_residual_subspace() -> None:
    converter = _load_converter()
    curve = torch.ones(5, 1)
    shared_residual = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = {
        "time_delta": torch.cat([shared_residual, 2 * shared_residual], dim=1),
        "block_0": -3 * shared_residual,
    }

    extended, diagnostics = converter.derive_extended_curve_basis(
        curve_coordinates=curve,
        feature_groups=targets,
        max_relative_error=1e-6,
        max_extra_dimensions=2,
    )

    torch.testing.assert_close(extended[:, :1], curve)
    assert extended.shape == (5, 2)
    assert diagnostics["extra_dimensions"] == 1
    assert diagnostics["baseline_max_relative_error"] == pytest.approx(1.0)
    assert diagnostics["final_max_relative_error"] < 1e-6


def test_basis_derivation_fails_when_dimension_budget_is_insufficient() -> None:
    converter = _load_converter()

    with pytest.raises(ValueError, match="requires more than 0 extra dimensions"):
        converter.derive_extended_curve_basis(
            curve_coordinates=torch.ones(4, 1),
            feature_groups={"delta": torch.tensor([[0.0], [1.0], [0.0], [-1.0]])},
            max_relative_error=1e-6,
            max_extra_dimensions=0,
        )


def test_curve_extension_model_reuses_shards_and_overlays_only_the_table(
    tmp_path: Path,
) -> None:
    converter = _load_converter()
    curve = tmp_path / "curve"
    destination = tmp_path / "curve-extended"
    base_table = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32
    )
    _write_model(
        curve,
        {
            "adaln_t_table": base_table,
            "blocks.0.adaln_proj.linear.weight": torch.ones(4, 2),
            "blocks.0.adaln_proj.linear.bias": torch.zeros(4),
        },
        {"adaln_curve_grid": 3, "time_embed_dim": 2},
    )
    (curve / "vae").mkdir()
    (curve / "vae" / "config.json").write_text("{}\n")
    extended = torch.cat(
        [base_table, torch.tensor([[-1.0], [0.0], [1.0]])], dim=1
    )

    result = converter.prepare_curve_extension_model(
        curve_model=curve,
        destination=destination,
        extended_coordinates=extended,
    )

    config = json.loads((destination / "transformer" / "config.json").read_text())
    assert config["adaln_curve_grid"] == 3
    assert config["adaln_curve_base_dim"] == 2
    assert config["time_embed_dim"] == 3
    assert (destination / "vae").is_symlink()
    source_shard = "model-00001-of-00001.safetensors"
    assert (destination / "transformer" / source_shard).is_symlink()
    index = json.loads(
        (destination / "transformer" / "model.safetensors.index.json").read_text()
    )
    overlay_name = index["weight_map"]["adaln_t_table"]
    assert overlay_name != source_shard
    with safe_open(
        destination / "transformer" / overlay_name,
        framework="pt",
        device="cpu",
    ) as overlay:
        torch.testing.assert_close(overlay.get_tensor("adaln_t_table"), extended)
    assert result["base_curve_dimensions"] == 2
    assert result["extended_curve_dimensions"] == 3


def test_converter_emits_curve_compatible_residual_adapter(tmp_path: Path) -> None:
    converter = _load_converter()
    torch.manual_seed(7)
    full = tmp_path / "full"
    curve = tmp_path / "curve"
    adapter = tmp_path / "adapter"
    destination = tmp_path / "derived"
    adapter.mkdir()

    full_tensors = {
        "time_embedder.proj_in.weight": torch.randn(5, 4),
        "time_embedder.proj_in.bias": torch.randn(5),
        "time_embedder.proj_out.weight": torch.randn(3, 5),
        "time_embedder.proj_out.bias": torch.randn(3),
        "blocks.0.adaln_proj.linear.weight": torch.randn(6, 3),
        "blocks.0.adaln_proj.linear.bias": torch.randn(6),
    }
    _write_model(
        full,
        full_tensors,
        {"timestep_input_dim": 4, "time_embed_dim": 3},
    )
    _write_model(
        curve,
        {
            "adaln_t_table": torch.randn(9, 2),
            "blocks.0.adaln_proj.linear.weight": torch.randn(6, 2),
            "blocks.0.adaln_proj.linear.bias": torch.randn(6),
        },
        {"timestep_input_dim": 4, "time_embed_dim": 2, "adaln_curve_grid": 9},
    )

    rank_pattern = {
        "time_embedder.proj_in": 2,
        "time_embedder.proj_out": 2,
        "blocks.0.adaln_proj.linear": 2,
        "video_patch_proj": 2,
    }
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2,
                "rank_pattern": rank_pattern,
                "alpha_pattern": rank_pattern,
                "target_modules": ["linear", "proj_in", "proj_out"],
            }
        )
        + "\n"
    )
    adapter_tensors = {
        "time_embedder.proj_in.lora_A.weight": torch.randn(2, 4),
        "time_embedder.proj_in.lora_B.weight": torch.randn(5, 2),
        "time_embedder.proj_out.lora_A.weight": torch.randn(2, 5),
        "time_embedder.proj_out.lora_B.weight": torch.randn(3, 2),
        "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(2, 3),
        "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(6, 2),
        "video_patch_proj.lora_A.weight": torch.randn(2, 4),
        "video_patch_proj.lora_B.weight": torch.randn(5, 2),
    }
    save_file(adapter_tensors, adapter / "adapter_model.safetensors")
    source_provenance = {
        "source_repository": "example/Tutu-MiniMax-H3",
        "source_revision": "0123456789abcdef",
        "source_config_sha256": "a" * 64,
        "source_weights_sha256": "b" * 64,
        "prepared_weights_sha256": converter._sha256(
            adapter / "adapter_model.safetensors"
        ),
    }
    (adapter / "tradecraft-manifest.json").write_text(
        json.dumps(source_provenance) + "\n"
    )

    result = converter.prepare_curve_tutu_adapter(
        full_model=full,
        curve_model=curve,
        source_adapter=adapter,
        destination=destination,
        adapter_scale=0.8,
        max_basis_relative_error=2.0,
        device="cpu",
    )

    assert result["status"] == "derived_unqualified"
    assert result["requires_checkpoint_format"] == "curve_adaln"
    assert result["removed_modules"] == [
        "time_embedder.proj_in",
        "time_embedder.proj_out",
    ]
    config = json.loads((destination / "adapter_config.json").read_text())
    assert "time_embedder.proj_in" not in config["rank_pattern"]
    assert config["rank_pattern"]["blocks.0.adaln_proj.linear"] == 2
    assert config["alpha_pattern"]["blocks.0.adaln_proj.linear"] == 2
    assert "proj_in" not in config["target_modules"]
    with safe_open(
        destination / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as converted:
        assert "time_embedder.proj_in.lora_A.weight" not in converted.keys()
        torch.testing.assert_close(
            converted.get_tensor(
                "blocks.0.adaln_proj.linear.lora_A.weight"
            ),
            torch.eye(2),
        )
        torch.testing.assert_close(
            converted.get_tensor("video_patch_proj.lora_A.weight"),
            adapter_tensors["video_patch_proj.lora_A.weight"],
        )
    manifest = json.loads((destination / "tradecraft-manifest.json").read_text())
    assert manifest == result
    assert manifest["adaln_module_count"] == 1
    assert manifest["source_adapter_scale"] == 0.8

    bundle = tmp_path / "extended-bundle"
    bundle_result = converter.prepare_extended_curve_tutu_bundle(
        full_model=full,
        curve_model=curve,
        source_adapter=adapter,
        destination=bundle,
        adapter_scale=0.8,
        max_basis_relative_error=0.01,
        max_extra_dimensions=7,
        device="cpu",
    )
    bundle_config = json.loads(
        (bundle / "model" / "transformer" / "config.json").read_text()
    )
    adapter_config = json.loads(
        (bundle / "adapter" / "adapter_config.json").read_text()
    )
    assert bundle_result["status"] == "derived_unqualified"
    assert bundle_result["source_adapter_provenance"] == source_provenance
    assert bundle_config["adaln_curve_base_dim"] == 2
    assert bundle_config["time_embed_dim"] > 2
    assert (
        adapter_config["rank_pattern"]["blocks.0.adaln_proj.linear"]
        == bundle_config["time_embed_dim"]
    )


def test_converter_fails_closed_when_curve_basis_error_exceeds_limit(
    tmp_path: Path,
) -> None:
    converter = _load_converter()
    full = tmp_path / "full"
    curve = tmp_path / "curve"
    adapter = tmp_path / "adapter"
    destination = tmp_path / "derived"
    adapter.mkdir()
    _write_model(
        full,
        {
            "time_embedder.proj_in.weight": torch.eye(2),
            "time_embedder.proj_in.bias": torch.zeros(2),
            "time_embedder.proj_out.weight": torch.eye(2),
            "time_embedder.proj_out.bias": torch.zeros(2),
            "blocks.0.adaln_proj.linear.weight": torch.eye(2),
            "blocks.0.adaln_proj.linear.bias": torch.zeros(2),
        },
        {"timestep_input_dim": 2, "time_embed_dim": 2},
    )
    _write_model(
        curve,
        {
            "adaln_t_table": torch.ones(5, 1),
            "blocks.0.adaln_proj.linear.weight": torch.ones(2, 1),
            "blocks.0.adaln_proj.linear.bias": torch.zeros(2),
        },
        {"timestep_input_dim": 2, "time_embed_dim": 1, "adaln_curve_grid": 5},
    )
    patterns = {
        "time_embedder.proj_in": 1,
        "time_embedder.proj_out": 1,
        "blocks.0.adaln_proj.linear": 1,
    }
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "lora_alpha": 1,
                "rank_pattern": patterns,
                "alpha_pattern": patterns,
                "target_modules": ["linear", "proj_in", "proj_out"],
            }
        )
        + "\n"
    )
    save_file(
        {
            "time_embedder.proj_in.lora_A.weight": torch.ones(1, 2),
            "time_embedder.proj_in.lora_B.weight": torch.ones(2, 1),
            "time_embedder.proj_out.lora_A.weight": torch.ones(1, 2),
            "time_embedder.proj_out.lora_B.weight": torch.ones(2, 1),
            "blocks.0.adaln_proj.linear.lora_A.weight": torch.ones(1, 2),
            "blocks.0.adaln_proj.linear.lora_B.weight": torch.ones(2, 1),
        },
        adapter / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="curve basis relative error"):
        converter.prepare_curve_tutu_adapter(
            full_model=full,
            curve_model=curve,
            source_adapter=adapter,
            destination=destination,
            adapter_scale=0.8,
            max_basis_relative_error=0.0,
            device="cpu",
        )

    assert not destination.exists()
