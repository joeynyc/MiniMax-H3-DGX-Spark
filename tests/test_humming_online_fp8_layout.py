import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.humming import (
    _configure_humming_online_fp8_compute,
    _online_fp8_input_quant_config,
    _restore_canonical_online_fp8_weight_layout,
)


def _layer(weight: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        output_partition_sizes=[3],
        input_size_per_partition=4,
    )


def test_online_fp8_transpose_is_restored_for_humming() -> None:
    canonical = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn)
    layer = _layer(canonical.t())

    _restore_canonical_online_fp8_weight_layout(layer)

    assert layer.weight.is_contiguous()
    assert tuple(layer.weight.shape) == (3, 4)
    assert torch.equal(layer.weight.float(), canonical.float())


def test_unexpected_online_fp8_shape_is_rejected() -> None:
    layer = _layer(torch.zeros((2, 4), dtype=torch.float8_e4m3fn))

    with pytest.raises(ValueError, match="weight layout mismatch"):
        _restore_canonical_online_fp8_weight_layout(layer)


def test_online_fp8_activation_config_preserves_per_token_scaling() -> None:
    group_shape = SimpleNamespace(
        is_per_token=lambda: True,
        is_per_group=lambda: False,
    )
    config = SimpleNamespace(
        activation_quant_key=SimpleNamespace(
            scale=SimpleNamespace(static=False, group_shape=group_shape)
        )
    )

    assert _online_fp8_input_quant_config(config) == {
        "quant_method": "humming",
        "dtype": "float8e4m3",
        "group_size": 0,
    }


def test_humming_batch_invariant_is_scoped_to_compute_config(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HUMMING_BATCH_INVARIANT", "1")
    layer = SimpleNamespace(
        compute_config=json.dumps(
            {
                "use_batch_invariant": False,
                "use_f16_accum": False,
                "gemm_type": "dense",
            }
        )
    )

    _configure_humming_online_fp8_compute(layer)

    assert json.loads(layer.compute_config)["use_batch_invariant"] is True
