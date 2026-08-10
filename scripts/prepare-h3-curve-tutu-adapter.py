#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Project a full-AdaLN H3 LoRA into an existing curve-AdaLN basis.

The converter is deliberately local-only: it accepts operator-supplied model
and adapter directories, performs no network access, and emits an unqualified
residual adapter. Hardware and human review remain separate promotion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file

_TIME_MODULES = ("time_embedder.proj_in", "time_embedder.proj_out")
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


class _ShardedTensorSource:
    def __init__(self, model: Path) -> None:
        self.transformer = model / "transformer"
        self.config_path = self.transformer / "config.json"
        self.index_path = self.transformer / "model.safetensors.index.json"
        if not self.config_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(
                f"model does not contain an indexed transformer: {model}"
            )
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, Mapping):
            raise ValueError(f"invalid transformer index: {self.index_path}")
        self.weight_map = {str(name): str(shard) for name, shard in raw_weight_map.items()}

    def tensor(self, name: str, *, device: torch.device) -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise ValueError(f"transformer tensor is missing: {name}")
        shard = self.transformer / shard_name
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            value = checkpoint.get_tensor(name)
        return value.to(device=device, dtype=torch.float32)


class _ProjectionInputs:
    __slots__ = (
        "full",
        "curve",
        "config_path",
        "weights_path",
        "adapter_config",
        "source_provenance",
        "ranks",
        "alphas",
        "adapter_tensors",
        "adaln_modules",
        "coordinates",
        "base_features",
        "adapted_features",
    )

    def __init__(
        self,
        *,
        full: _ShardedTensorSource,
        curve: _ShardedTensorSource,
        config_path: Path,
        weights_path: Path,
        adapter_config: dict[str, Any],
        source_provenance: dict[str, Any] | None,
        ranks: dict[str, int],
        alphas: dict[str, int],
        adapter_tensors: dict[str, torch.Tensor],
        adaln_modules: list[str],
        coordinates: torch.Tensor,
        base_features: torch.Tensor,
        adapted_features: torch.Tensor,
    ) -> None:
        self.full = full
        self.curve = curve
        self.config_path = config_path
        self.weights_path = weights_path
        self.adapter_config = adapter_config
        self.source_provenance = source_provenance
        self.ranks = ranks
        self.alphas = alphas
        self.adapter_tensors = adapter_tensors
        self.adaln_modules = adaln_modules
        self.coordinates = coordinates
        self.base_features = base_features
        self.adapted_features = adapted_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_pattern(config: Mapping[str, Any], field: str) -> dict[str, int]:
    raw = config.get(field)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"adapter config requires a non-empty {field}")
    result: dict[str, int] = {}
    for name, value in raw.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} values must be positive integers")
        result[str(name)] = value
    return result


def _module_name(tensor_name: str) -> str:
    for suffix in _LORA_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    raise ValueError(f"unsupported adapter tensor: {tensor_name}")


def _linear_with_lora(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return functional.linear(inputs, weight, bias) + scale * functional.linear(
        functional.linear(inputs, lora_a), lora_b
    )


def _relative_projection_error(
    coordinates: torch.Tensor,
    pseudoinverse: torch.Tensor,
    values: torch.Tensor,
) -> float:
    residual = values - coordinates @ (pseudoinverse @ values)
    denominator = torch.linalg.vector_norm(values)
    numerator = torch.linalg.vector_norm(residual)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else math.inf
    return float((numerator / denominator).item())


def derive_extended_curve_basis(
    *,
    curve_coordinates: torch.Tensor,
    feature_groups: Mapping[str, torch.Tensor],
    max_relative_error: float,
    max_extra_dimensions: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extend a curve basis with the smallest shared residual SVD subspace."""
    if curve_coordinates.ndim != 2 or curve_coordinates.shape[0] < 2:
        raise ValueError("curve coordinates must be [grid >= 2, dimensions]")
    if not feature_groups:
        raise ValueError("at least one feature group is required")
    if not math.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError("maximum relative error must be finite and non-negative")
    if (
        not isinstance(max_extra_dimensions, int)
        or isinstance(max_extra_dimensions, bool)
        or max_extra_dimensions < 0
    ):
        raise ValueError("maximum extra dimensions must be a non-negative integer")

    coordinates = curve_coordinates.to(torch.float32)
    base_pseudoinverse = torch.linalg.pinv(coordinates.to(torch.float64)).to(
        dtype=coordinates.dtype
    )
    normalized_residuals: list[torch.Tensor] = []
    baseline_errors: dict[str, float] = {}
    normalized_groups: dict[str, torch.Tensor] = {}
    for name, raw_values in feature_groups.items():
        values = raw_values.to(device=coordinates.device, dtype=coordinates.dtype)
        if values.ndim != 2 or values.shape[0] != coordinates.shape[0]:
            raise ValueError(f"feature group {name} must share the curve grid")
        normalized_groups[str(name)] = values
        error = _relative_projection_error(coordinates, base_pseudoinverse, values)
        baseline_errors[str(name)] = error
        norm = torch.linalg.vector_norm(values)
        if norm.item() != 0:
            residual = values - coordinates @ (base_pseudoinverse @ values)
            normalized_residuals.append(residual / norm)

    baseline_max = max(baseline_errors.values())
    if baseline_max <= max_relative_error:
        return coordinates.clone(), {
            "extra_dimensions": 0,
            "baseline_max_relative_error": baseline_max,
            "final_max_relative_error": baseline_max,
            "baseline_relative_errors": baseline_errors,
            "final_relative_errors": baseline_errors,
        }
    if not normalized_residuals:
        raise ValueError("non-zero curve basis error has no residual feature data")
    residual_matrix = torch.cat(normalized_residuals, dim=1)
    left_vectors = torch.linalg.svd(
        residual_matrix, full_matrices=False
    ).U
    available = min(max_extra_dimensions, left_vectors.shape[1])
    for extra_dimensions in range(1, available + 1):
        extended = torch.cat(
            [coordinates, left_vectors[:, :extra_dimensions]], dim=1
        )
        pseudoinverse = torch.linalg.pinv(extended.to(torch.float64)).to(
            dtype=extended.dtype
        )
        final_errors = {
            name: _relative_projection_error(extended, pseudoinverse, values)
            for name, values in normalized_groups.items()
        }
        final_max = max(final_errors.values())
        if final_max <= max_relative_error:
            return extended, {
                "extra_dimensions": extra_dimensions,
                "baseline_max_relative_error": baseline_max,
                "final_max_relative_error": final_max,
                "baseline_relative_errors": baseline_errors,
                "final_relative_errors": final_errors,
            }
    raise ValueError(
        "curve basis requires more than "
        f"{max_extra_dimensions} extra dimensions to reach relative error "
        f"{max_relative_error:.8f}"
    )


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(
        os.path.relpath(source, start=destination.parent),
        target_is_directory=source.is_dir(),
    )


def prepare_curve_extension_model(
    *,
    curve_model: str | Path,
    destination: str | Path,
    extended_coordinates: torch.Tensor,
) -> dict[str, Any]:
    """Create a symlinked curve model with one small table overlay shard."""
    curve_root = Path(curve_model).expanduser().resolve(strict=True)
    output_root = Path(destination).expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"curve model destination already exists: {output_root}")
    source = _ShardedTensorSource(curve_root)
    base_grid = int(source.config.get("adaln_curve_grid", 0))
    base_dimensions = int(source.config.get("time_embed_dim", 0))
    coordinates = extended_coordinates.detach().to(dtype=torch.float32, device="cpu")
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != base_grid
        or coordinates.shape[1] <= base_dimensions
    ):
        raise ValueError(
            "extended coordinates must preserve the grid and add dimensions"
        )
    base_table = source.tensor("adaln_t_table", device=torch.device("cpu"))
    if not torch.equal(coordinates[:, :base_dimensions], base_table):
        raise ValueError("extended coordinates must preserve the base curve table")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for child in curve_root.iterdir():
            if child.name != "transformer":
                _relative_symlink(child, temporary / child.name)
        output_transformer = temporary / "transformer"
        output_transformer.mkdir()
        config = dict(source.config)
        config.update(
            {
                "adaln_curve_base_dim": base_dimensions,
                "time_embed_dim": int(coordinates.shape[1]),
            }
        )
        (output_transformer / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_index = json.loads(source.index_path.read_text(encoding="utf-8"))
        weight_map = {
            str(name): str(shard)
            for name, shard in raw_index["weight_map"].items()
        }
        for shard_name in sorted(set(weight_map.values())):
            _relative_symlink(
                source.transformer / shard_name,
                output_transformer / shard_name,
            )
        overlay_name = "model-tradecraft-curve-extension.safetensors"
        overlay_path = output_transformer / overlay_name
        save_file(
            {"adaln_t_table": coordinates.contiguous()},
            overlay_path,
            metadata={"conversion": "tradecraft-h3-curve-extension-v1"},
        )
        weight_map["adaln_t_table"] = overlay_name
        metadata = dict(raw_index.get("metadata") or {})
        total_size = metadata.get("total_size")
        if isinstance(total_size, int) and not isinstance(total_size, bool):
            metadata["total_size"] = (
                total_size
                - base_table.numel() * base_table.element_size()
                + coordinates.numel() * coordinates.element_size()
            )
        output_index = {
            "metadata": metadata,
            "weight_map": dict(sorted(weight_map.items())),
        }
        index_path = output_transformer / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps(output_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "format_version": "tradecraft-h3-curve-extension-v1",
            "status": "derived_unqualified",
            "curve_grid": base_grid,
            "base_curve_dimensions": base_dimensions,
            "extended_curve_dimensions": int(coordinates.shape[1]),
            "source_model_index_sha256": _sha256(source.index_path),
            "overlay_sha256": _sha256(overlay_path),
            "extended_model_index_sha256": _sha256(index_path),
        }
        os.replace(temporary, output_root)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def project_adaln_delta(
    *,
    curve_coordinates: torch.Tensor,
    base_features: torch.Tensor,
    adapted_features: torch.Tensor,
    base_weight: torch.Tensor,
    source_lora_a: torch.Tensor,
    source_lora_b: torch.Tensor,
    source_lora_scale: float,
    output_adapter_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return a rank-D LoRA reproducing a full-AdaLN adapter in curve space.

    ``base_features`` and ``adapted_features`` are the post-SiLU full timestep
    features. The output adapter uses an identity A matrix over the D curve
    coordinates and a fitted B matrix. The fit is computed in feature space,
    so no grid-by-AdaLN-output tensor is materialized.
    """
    if curve_coordinates.ndim != 2 or curve_coordinates.shape[0] < 2:
        raise ValueError("curve coordinates must be [grid >= 2, dimensions]")
    if base_features.shape != adapted_features.shape:
        raise ValueError("base and adapted timestep features must have equal shape")
    if base_features.ndim != 2 or base_features.shape[0] != curve_coordinates.shape[0]:
        raise ValueError("timestep features must share the curve grid")
    if not math.isfinite(output_adapter_scale) or output_adapter_scale <= 0:
        raise ValueError("output adapter scale must be finite and positive")
    if source_lora_a.shape[1] != base_features.shape[1]:
        raise ValueError("source AdaLN LoRA input does not match full time features")
    if source_lora_b.shape != (base_weight.shape[0], source_lora_a.shape[0]):
        raise ValueError("source AdaLN LoRA A/B shapes do not match the base weight")

    coordinate_dtype = curve_coordinates.dtype
    pseudoinverse = torch.linalg.pinv(curve_coordinates.to(torch.float64)).to(
        dtype=coordinate_dtype
    )
    time_delta = adapted_features - base_features
    lora_features = adapted_features @ source_lora_a.T
    time_coefficients = pseudoinverse @ time_delta
    lora_coefficients = pseudoinverse @ lora_features
    curve_b = (
        base_weight @ time_coefficients.T
        + source_lora_scale * (source_lora_b @ lora_coefficients.T)
    ) / output_adapter_scale
    dimensions = curve_coordinates.shape[1]
    curve_a = torch.eye(
        dimensions,
        dtype=curve_coordinates.dtype,
        device=curve_coordinates.device,
    )
    diagnostics = {
        "time_feature_relative_error": _relative_projection_error(
            curve_coordinates, pseudoinverse, time_delta
        ),
        "lora_feature_relative_error": _relative_projection_error(
            curve_coordinates, pseudoinverse, lora_features
        ),
    }
    return curve_a, curve_b, diagnostics


def _full_time_features(
    *,
    full: _ShardedTensorSource,
    adapter_tensors: Mapping[str, torch.Tensor],
    ranks: Mapping[str, int],
    alphas: Mapping[str, int],
    adapter_scale: float,
    grid: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dimensions = int(full.config.get("timestep_input_dim", 0))
    if input_dimensions <= 0 or input_dimensions % 2:
        raise ValueError("full checkpoint timestep_input_dim must be positive and even")
    timesteps = torch.linspace(0.0, 1.0, grid, dtype=torch.float32, device=device)
    half = input_dimensions // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    arguments = timesteps[:, None] * frequencies[None]
    frequency_features = torch.cat(
        [torch.cos(arguments), torch.sin(arguments)], dim=-1
    )

    def adapter_tensor(name: str) -> torch.Tensor:
        value = adapter_tensors.get(name)
        if value is None:
            raise ValueError(f"adapter tensor is missing: {name}")
        return value.to(device=device, dtype=torch.float32)

    in_name, out_name = _TIME_MODULES
    in_scale = adapter_scale * alphas[in_name] / ranks[in_name]
    out_scale = adapter_scale * alphas[out_name] / ranks[out_name]
    in_weight = full.tensor(f"{in_name}.weight", device=device)
    in_bias = full.tensor(f"{in_name}.bias", device=device)
    out_weight = full.tensor(f"{out_name}.weight", device=device)
    out_bias = full.tensor(f"{out_name}.bias", device=device)
    base_hidden = functional.silu(functional.linear(frequency_features, in_weight, in_bias))
    base_time = functional.linear(base_hidden, out_weight, out_bias)
    adapted_hidden = functional.silu(
        _linear_with_lora(
            frequency_features,
            in_weight,
            in_bias,
            adapter_tensor(f"{in_name}.lora_A.weight"),
            adapter_tensor(f"{in_name}.lora_B.weight"),
            in_scale,
        )
    )
    adapted_time = _linear_with_lora(
        adapted_hidden,
        out_weight,
        out_bias,
        adapter_tensor(f"{out_name}.lora_A.weight"),
        adapter_tensor(f"{out_name}.lora_B.weight"),
        out_scale,
    )
    return functional.silu(base_time), functional.silu(adapted_time)


def _load_projection_inputs(
    *,
    full_model: str | Path,
    curve_model: str | Path,
    source_adapter: str | Path,
    adapter_scale: float,
    device: torch.device,
) -> _ProjectionInputs:
    full_root = Path(full_model).expanduser().resolve(strict=True)
    curve_root = Path(curve_model).expanduser().resolve(strict=True)
    adapter_root = Path(source_adapter).expanduser().resolve(strict=True)
    config_path = adapter_root / "adapter_config.json"
    weights_path = adapter_root / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "source adapter requires adapter_config.json and adapter_model.safetensors"
        )

    full = _ShardedTensorSource(full_root)
    curve = _ShardedTensorSource(curve_root)
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_provenance: dict[str, Any] | None = None
    source_manifest_path = adapter_root / "tradecraft-manifest.json"
    if source_manifest_path.is_file():
        raw_provenance = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_provenance, Mapping):
            raise ValueError("source adapter manifest must be a JSON object")
        required_provenance = (
            "source_repository",
            "source_revision",
            "source_config_sha256",
            "source_weights_sha256",
            "prepared_weights_sha256",
        )
        if any(
            not str(raw_provenance.get(name, "")).strip()
            for name in required_provenance
        ):
            raise ValueError("source adapter manifest has incomplete provenance")
        if raw_provenance["prepared_weights_sha256"] != _sha256(weights_path):
            raise ValueError("source adapter manifest does not match prepared weights")
        source_provenance = dict(raw_provenance)
    ranks = _positive_pattern(adapter_config, "rank_pattern")
    alphas = _positive_pattern(adapter_config, "alpha_pattern")
    if set(ranks) != set(alphas):
        raise ValueError("adapter rank and alpha patterns must target equal modules")
    missing_time = [name for name in _TIME_MODULES if name not in ranks]
    if missing_time:
        raise ValueError(
            "source adapter lacks full timestep modules: " + ", ".join(missing_time)
        )
    adaln_modules = sorted(
        name for name in ranks if ".adaln_proj.linear" in name
    )
    if not adaln_modules:
        raise ValueError("source adapter has no AdaLN projection modules")

    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        adapter_tensors = {
            name: checkpoint.get_tensor(name).clone() for name in checkpoint.keys()
        }
    expected_names = {
        f"{name}{suffix}" for name in ranks for suffix in _LORA_SUFFIXES
    }
    if set(adapter_tensors) != expected_names:
        missing = sorted(expected_names - set(adapter_tensors))
        extra = sorted(set(adapter_tensors) - expected_names)
        raise ValueError(
            "adapter tensors do not match config; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    curve_grid = int(curve.config.get("adaln_curve_grid", 0))
    curve_dimensions = int(curve.config.get("time_embed_dim", 0))
    coordinates = curve.tensor("adaln_t_table", device=device)
    if coordinates.shape != (curve_grid, curve_dimensions) or curve_grid < 2:
        raise ValueError("curve checkpoint metadata does not match adaln_t_table")
    base_features, adapted_features = _full_time_features(
        full=full,
        adapter_tensors=adapter_tensors,
        ranks=ranks,
        alphas=alphas,
        adapter_scale=adapter_scale,
        grid=curve_grid,
        device=device,
    )
    if base_features.shape[1] != int(full.config.get("time_embed_dim", 0)):
        raise ValueError("full checkpoint time embedding metadata does not match weights")
    return _ProjectionInputs(
        full=full,
        curve=curve,
        config_path=config_path,
        weights_path=weights_path,
        adapter_config=adapter_config,
        source_provenance=source_provenance,
        ranks=ranks,
        alphas=alphas,
        adapter_tensors=adapter_tensors,
        adaln_modules=adaln_modules,
        coordinates=coordinates,
        base_features=base_features,
        adapted_features=adapted_features,
    )


def analyze_curve_basis_requirements(
    *,
    full_model: str | Path,
    curve_model: str | Path,
    source_adapter: str | Path,
    adapter_scale: float,
    max_basis_relative_error: float,
    max_extra_dimensions: int,
    device: str,
) -> dict[str, Any]:
    """Measure the smallest joint curve basis needed by one full-AdaLN LoRA."""
    if not math.isfinite(adapter_scale) or adapter_scale <= 0:
        raise ValueError("adapter scale must be finite and positive")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    inputs = _load_projection_inputs(
        full_model=full_model,
        curve_model=curve_model,
        source_adapter=source_adapter,
        adapter_scale=adapter_scale,
        device=compute_device,
    )
    feature_groups = _adapter_basis_feature_groups(inputs, device=compute_device)
    extended, diagnostics = derive_extended_curve_basis(
        curve_coordinates=inputs.coordinates,
        feature_groups=feature_groups,
        max_relative_error=max_basis_relative_error,
        max_extra_dimensions=max_extra_dimensions,
    )
    result = {
        "format_version": "tradecraft-h3-curve-basis-analysis-v1",
        "status": "analyzed_unqualified",
        "source_adapter_scale": adapter_scale,
        "source_adapter_config_sha256": _sha256(inputs.config_path),
        "source_adapter_weights_sha256": _sha256(inputs.weights_path),
        "full_model_index_sha256": _sha256(inputs.full.index_path),
        "curve_model_index_sha256": _sha256(inputs.curve.index_path),
        "curve_grid": int(extended.shape[0]),
        "base_curve_dimensions": int(inputs.coordinates.shape[1]),
        "required_curve_dimensions": int(extended.shape[1]),
        "adaln_module_count": len(inputs.adaln_modules),
        "maximum_allowed_basis_relative_error": max_basis_relative_error,
        "basis_diagnostics": diagnostics,
    }
    if inputs.source_provenance is not None:
        result["source_adapter_provenance"] = inputs.source_provenance
    return result


def _adapter_basis_feature_groups(
    inputs: _ProjectionInputs,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    groups = {
        "full_timestep_delta": inputs.adapted_features - inputs.base_features
    }
    for module_name in inputs.adaln_modules:
        source_a = inputs.adapter_tensors[
            f"{module_name}.lora_A.weight"
        ].to(device=device, dtype=torch.float32)
        groups[f"{module_name}.lora_input"] = inputs.adapted_features @ source_a.T
    return groups


def _project_residual_adapter(
    *,
    inputs: _ProjectionInputs,
    coordinates: torch.Tensor,
    adapter_scale: float,
    max_basis_relative_error: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    ranks = dict(inputs.ranks)
    alphas = dict(inputs.alphas)
    converted = {
        name: tensor
        for name, tensor in inputs.adapter_tensors.items()
        if _module_name(name)
        not in set(_TIME_MODULES) | set(inputs.adaln_modules)
    }
    module_diagnostics: dict[str, dict[str, float]] = {}
    maximum_error = 0.0
    dimensions = int(coordinates.shape[1])
    base_dimensions = int(inputs.coordinates.shape[1])
    for module_name in inputs.adaln_modules:
        full_weight = inputs.full.tensor(f"{module_name}.weight", device=device)
        curve_weight = inputs.curve.tensor(f"{module_name}.weight", device=device)
        if curve_weight.shape != (full_weight.shape[0], base_dimensions):
            raise ValueError(f"curve AdaLN weight shape does not match {module_name}")
        source_a = inputs.adapter_tensors[f"{module_name}.lora_A.weight"].to(
            device=device, dtype=torch.float32
        )
        source_b = inputs.adapter_tensors[f"{module_name}.lora_B.weight"].to(
            device=device, dtype=torch.float32
        )
        source_scale = adapter_scale * alphas[module_name] / ranks[module_name]
        curve_a, curve_b, diagnostics = project_adaln_delta(
            curve_coordinates=coordinates,
            base_features=inputs.base_features,
            adapted_features=inputs.adapted_features,
            base_weight=full_weight,
            source_lora_a=source_a,
            source_lora_b=source_b,
            source_lora_scale=source_scale,
            output_adapter_scale=adapter_scale,
        )
        module_error = max(diagnostics.values())
        maximum_error = max(maximum_error, module_error)
        module_diagnostics[module_name] = diagnostics
        converted[f"{module_name}.lora_A.weight"] = curve_a.cpu()
        converted[f"{module_name}.lora_B.weight"] = curve_b.cpu()
        ranks[module_name] = dimensions
        alphas[module_name] = dimensions
        del full_weight, curve_weight, source_a, source_b, curve_a, curve_b

    if maximum_error > max_basis_relative_error:
        raise ValueError(
            "curve basis relative error "
            f"{maximum_error:.8f} exceeds limit {max_basis_relative_error:.8f}"
        )
    for module_name in _TIME_MODULES:
        ranks.pop(module_name)
        alphas.pop(module_name)
    converted_config = dict(inputs.adapter_config)
    converted_config.update(
        {
            "rank_pattern": dict(sorted(ranks.items())),
            "alpha_pattern": dict(sorted(alphas.items())),
            "target_modules": sorted(
                {name.rsplit(".", 1)[-1] for name in ranks}
            ),
        }
    )
    diagnostics = {
        "maximum_basis_relative_error": maximum_error,
        "basis_diagnostics": module_diagnostics,
        "residual_module_count": len(ranks),
    }
    return converted, converted_config, diagnostics


def prepare_extended_curve_tutu_bundle(
    *,
    full_model: str | Path,
    curve_model: str | Path,
    source_adapter: str | Path,
    destination: str | Path,
    adapter_scale: float,
    max_basis_relative_error: float,
    max_extra_dimensions: int,
    device: str,
) -> dict[str, Any]:
    """Emit a tiny curve-model overlay and its matching residual adapter."""
    if not math.isfinite(adapter_scale) or adapter_scale <= 0:
        raise ValueError("adapter scale must be finite and positive")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_root = Path(destination).expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"bundle destination already exists: {output_root}")
    inputs = _load_projection_inputs(
        full_model=full_model,
        curve_model=curve_model,
        source_adapter=source_adapter,
        adapter_scale=adapter_scale,
        device=compute_device,
    )
    extended, basis_diagnostics = derive_extended_curve_basis(
        curve_coordinates=inputs.coordinates,
        feature_groups=_adapter_basis_feature_groups(
            inputs,
            device=compute_device,
        ),
        max_relative_error=max_basis_relative_error,
        max_extra_dimensions=max_extra_dimensions,
    )
    if extended.shape[1] == inputs.coordinates.shape[1]:
        raise ValueError(
            "the source adapter already fits the base curve; use prepare instead"
        )
    converted, converted_config, projection = _project_residual_adapter(
        inputs=inputs,
        coordinates=extended,
        adapter_scale=adapter_scale,
        max_basis_relative_error=max_basis_relative_error,
        device=compute_device,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model_result = prepare_curve_extension_model(
            curve_model=curve_model,
            destination=temporary / "model",
            extended_coordinates=extended,
        )
        adapter_output = temporary / "adapter"
        adapter_output.mkdir()
        output_weights = adapter_output / "adapter_model.safetensors"
        save_file(
            converted,
            output_weights,
            metadata={
                "conversion": "tradecraft-h3-extended-curve-adaln-v1",
                "source_adapter_sha256": _sha256(inputs.weights_path),
            },
        )
        (adapter_output / "adapter_config.json").write_text(
            json.dumps(converted_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": "tradecraft-h3-extended-curve-bundle-v1",
            "status": "derived_unqualified",
            "requires_checkpoint_format": "extended_curve_adaln",
            "source_adapter_scale": adapter_scale,
            "source_adapter_config_sha256": _sha256(inputs.config_path),
            "source_adapter_weights_sha256": _sha256(inputs.weights_path),
            "full_model_index_sha256": _sha256(inputs.full.index_path),
            "curve_model_index_sha256": _sha256(inputs.curve.index_path),
            "curve_grid": int(extended.shape[0]),
            "base_curve_dimensions": int(inputs.coordinates.shape[1]),
            "extended_curve_dimensions": int(extended.shape[1]),
            "adaln_module_count": len(inputs.adaln_modules),
            "residual_module_count": projection["residual_module_count"],
            "removed_modules": list(_TIME_MODULES),
            "maximum_basis_relative_error": projection[
                "maximum_basis_relative_error"
            ],
            "maximum_allowed_basis_relative_error": max_basis_relative_error,
            "basis_derivation": basis_diagnostics,
            "basis_diagnostics": projection["basis_diagnostics"],
            "model_overlay": model_result,
            "prepared_weights_sha256": _sha256(output_weights),
        }
        if inputs.source_provenance is not None:
            manifest["source_adapter_provenance"] = inputs.source_provenance
        (adapter_output / "tradecraft-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "tradecraft-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_curve_tutu_adapter(
    *,
    full_model: str | Path,
    curve_model: str | Path,
    source_adapter: str | Path,
    destination: str | Path,
    adapter_scale: float,
    max_basis_relative_error: float,
    device: str,
) -> dict[str, Any]:
    """Build an unqualified curve-compatible residual from a full H3 LoRA."""
    if not math.isfinite(adapter_scale) or adapter_scale <= 0:
        raise ValueError("adapter scale must be finite and positive")
    if (
        not math.isfinite(max_basis_relative_error)
        or max_basis_relative_error < 0
    ):
        raise ValueError("maximum basis relative error must be finite and non-negative")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_root = Path(destination).expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"adapter destination already exists: {output_root}")
    inputs = _load_projection_inputs(
        full_model=full_model,
        curve_model=curve_model,
        source_adapter=source_adapter,
        adapter_scale=adapter_scale,
        device=compute_device,
    )
    curve_grid, curve_dimensions = inputs.coordinates.shape
    curve_pseudoinverse = torch.linalg.pinv(
        inputs.coordinates.to(torch.float64)
    ).to(
        dtype=inputs.coordinates.dtype
    )
    time_basis_error = _relative_projection_error(
        inputs.coordinates,
        curve_pseudoinverse,
        inputs.adapted_features - inputs.base_features,
    )
    if time_basis_error > max_basis_relative_error:
        raise ValueError(
            "curve basis relative error "
            f"{time_basis_error:.8f} exceeds limit "
            f"{max_basis_relative_error:.8f} for full timestep delta"
        )

    converted, converted_config, projection = _project_residual_adapter(
        inputs=inputs,
        coordinates=inputs.coordinates,
        adapter_scale=adapter_scale,
        max_basis_relative_error=max_basis_relative_error,
        device=compute_device,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        output_weights = temporary / "adapter_model.safetensors"
        save_file(
            converted,
            output_weights,
            metadata={
                "conversion": "tradecraft-h3-curve-adaln-v1",
                "source_adapter_sha256": _sha256(inputs.weights_path),
            },
        )
        (temporary / "adapter_config.json").write_text(
            json.dumps(converted_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": "tradecraft-h3-curve-adaln-v1",
            "status": "derived_unqualified",
            "requires_checkpoint_format": "curve_adaln",
            "source_adapter_scale": adapter_scale,
            "source_adapter_config_sha256": _sha256(inputs.config_path),
            "source_adapter_weights_sha256": _sha256(inputs.weights_path),
            "full_model_index_sha256": _sha256(inputs.full.index_path),
            "curve_model_index_sha256": _sha256(inputs.curve.index_path),
            "curve_grid": curve_grid,
            "curve_dimensions": curve_dimensions,
            "adaln_module_count": len(inputs.adaln_modules),
            "residual_module_count": projection["residual_module_count"],
            "removed_modules": list(_TIME_MODULES),
            "maximum_basis_relative_error": projection[
                "maximum_basis_relative_error"
            ],
            "maximum_allowed_basis_relative_error": max_basis_relative_error,
            "basis_diagnostics": projection["basis_diagnostics"],
            "prepared_weights_sha256": _sha256(output_weights),
        }
        if inputs.source_provenance is not None:
            manifest["source_adapter_provenance"] = inputs.source_provenance
        (temporary / "tradecraft-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--full-model", type=Path, required=True)
        command.add_argument("--curve-model", type=Path, required=True)
        command.add_argument("--source-adapter", type=Path, required=True)
        command.add_argument("--adapter-scale", type=float, default=0.8)
        command.add_argument(
            "--max-basis-relative-error", type=float, default=0.05
        )
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    analyze = commands.add_parser(
        "analyze-basis", help="measure the curve dimensions required by the adapter"
    )
    add_common(analyze)
    analyze.add_argument("--max-extra-dimensions", type=int, default=64)
    prepare = commands.add_parser(
        "prepare", help="emit an adapter when the existing curve basis already fits"
    )
    add_common(prepare)
    prepare.add_argument("--destination", type=Path, required=True)
    extended = commands.add_parser(
        "prepare-extended",
        help="emit a curve-extension model overlay and matching adapter",
    )
    add_common(extended)
    extended.add_argument("--max-extra-dimensions", type=int, default=64)
    extended.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        common = {
            "full_model": arguments.full_model,
            "curve_model": arguments.curve_model,
            "source_adapter": arguments.source_adapter,
            "adapter_scale": arguments.adapter_scale,
            "max_basis_relative_error": arguments.max_basis_relative_error,
            "device": arguments.device,
        }
        if arguments.command == "analyze-basis":
            result = analyze_curve_basis_requirements(
                **common,
                max_extra_dimensions=arguments.max_extra_dimensions,
            )
        elif arguments.command == "prepare-extended":
            result = prepare_extended_curve_tutu_bundle(
                **common,
                destination=arguments.destination,
                max_extra_dimensions=arguments.max_extra_dimensions,
            )
        else:
            result = prepare_curve_tutu_adapter(
                **common,
                destination=arguments.destination,
            )
    except Exception as error:  # noqa: BLE001 - operator CLI boundary
        print(f"error: curve adapter preparation failed: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
