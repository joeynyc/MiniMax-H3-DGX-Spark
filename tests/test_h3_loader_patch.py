# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
    _bind_native_cuda_fp8_activation_quantizers,
    _prepare_h3_checkpoint_weight,
)


def test_qkv_layout_is_normalized_before_parameter_loader():
    arch = MiniMaxH3DiTArchConfig(
        num_attention_heads=2,
        attention_head_dim=1,
    )
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    normalized = _prepare_h3_checkpoint_weight(
        "blocks.0.attn.qkv_proj.weight",
        grouped,
        arch=arch,
    )

    assert normalized[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_model_load_weights_preserves_parameter_loader_contract():
    calls = []
    param = torch.nn.Parameter(torch.empty(6, 1), requires_grad=False)

    def weight_loader(*args):
        calls.append(args)

    param.weight_loader = weight_loader

    class FakeModel:
        arch = MiniMaxH3DiTArchConfig(
            num_attention_heads=2,
            attention_head_dim=1,
        )

        @staticmethod
        def named_parameters():
            return [("blocks.0.attn.qkv_proj.weight", param)]

        @staticmethod
        def named_buffers():
            return []

    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    loaded = MiniMaxH3DiTModel.load_weights(
        FakeModel(),
        [("blocks.0.attn.qkv_proj.weight", grouped)],
    )

    assert loaded == {"blocks.0.attn.qkv_proj.weight"}
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert calls[0][0] is param
    assert calls[0][1][:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_fc1_uses_the_native_fused_loader_and_rejects_odd_rows():
    arch = MiniMaxH3DiTArchConfig()
    even = torch.empty(4, 2)
    assert (
        _prepare_h3_checkpoint_weight(
            "blocks.0.mlp.fc1.weight",
            even,
            arch=arch,
        )
        is even
    )

    odd = torch.empty(3, 2)
    try:
        _prepare_h3_checkpoint_weight(
            "blocks.0.mlp.fc1.weight",
            odd,
            arch=arch,
        )
    except ValueError as exc:
        assert "split evenly" in str(exc)
    else:
        raise AssertionError("odd fused fc1 rows must be rejected")


def test_fp8_activation_quantizers_bind_to_cuda_dispatch():
    root = torch.nn.Module()
    root.linear = torch.nn.Linear(2, 2)

    class FakeQuantizer:
        def __init__(self):
            self._forward_method = self.forward_native

        @staticmethod
        def forward_native(value):
            return ("native", value)

        @staticmethod
        def forward_cuda(value):
            return ("cuda", value)

    quantizer = FakeQuantizer()
    root.linear.quant_method = SimpleNamespace(
        fp8_linear=SimpleNamespace(quant_fp8=quantizer)
    )

    assert _bind_native_cuda_fp8_activation_quantizers(root) == 1
    assert quantizer._forward_method("probe") == ("cuda", "probe")


def test_adaln_keeps_activations_bf16_after_weight_quantization():
    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.empty(2, 2, dtype=torch.float8_e4m3fn),
            )
            self.input_dtype = None

        def forward(self, value):
            self.input_dtype = value.dtype
            return value, None

    proj = MiniMaxH3AdalnProj.__new__(MiniMaxH3AdalnProj)
    torch.nn.Module.__init__(proj)
    proj.linear = FakeLinear()
    proj.modality_num = 1
    proj.expand_ratio = 1
    proj.hidden_size = 2

    output = proj(torch.ones(1, 2, dtype=torch.float32))

    assert proj.linear.weight.dtype == torch.float8_e4m3fn
    assert proj.linear.input_dtype == torch.bfloat16
    assert output[0].dtype == torch.bfloat16
