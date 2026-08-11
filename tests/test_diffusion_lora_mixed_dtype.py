# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.lora.layers.base_linear import (
    DiffusionBaseLinearLayerWithLoRA,
)


@dataclass
class _DummyLoRAConfig:
    fully_sharded_loras: bool = False


class _DummyQuantMethod:
    def __init__(self, weight: torch.Tensor):
        self._weight = weight

    def apply(
        self,
        _base_layer,
        value: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output = value @ self._weight.t()
        return output if bias is None else output + bias


def test_diffusion_lora_casts_fp32_input_for_bf16_adapter() -> None:
    layer = DiffusionBaseLinearLayerWithLoRA.__new__(
        DiffusionBaseLinearLayerWithLoRA
    )
    layer.tp_size = 1
    layer.lora_config = _DummyLoRAConfig()
    layer.base_layer = type("Base", (), {})()
    layer.base_layer.quant_method = _DummyQuantMethod(
        torch.eye(2, dtype=torch.float32)
    )
    layer.lora_a_stacked = (
        torch.ones((1, 1, 1, 2), dtype=torch.bfloat16),
    )
    layer.lora_b_stacked = (
        torch.ones((1, 1, 2, 1), dtype=torch.bfloat16),
    )
    layer.output_slices = (2,)
    layer._diffusion_lora_active_slices = (True,)

    output = layer.apply(torch.tensor([[1.0, 2.0]], dtype=torch.float32))

    assert output.dtype is torch.float32
    torch.testing.assert_close(output, torch.tensor([[4.0, 5.0]]))
