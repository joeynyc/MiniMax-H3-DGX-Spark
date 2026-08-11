# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
    _bind_native_cuda_fp8_activation_quantizers,
    _interpolate_h3_adaln_curve,
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
    proj.apply_silu = True

    output = proj(torch.ones(1, 2, dtype=torch.float32))

    assert proj.linear.weight.dtype == torch.float8_e4m3fn
    assert proj.linear.input_dtype == torch.bfloat16
    assert output[0].dtype == torch.bfloat16


def test_curve_adaln_interpolates_normalized_timesteps_and_clamps_endpoints():
    table = torch.tensor(
        [
            [0.0, 10.0],
            [2.0, 12.0],
            [4.0, 14.0],
        ],
        dtype=torch.float32,
    )

    output = _interpolate_h3_adaln_curve(
        table,
        torch.tensor([-1.0, 0.25, 0.75, 2.0]),
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [3.0, 13.0],
                [4.0, 14.0],
            ]
        ),
    )


def test_curve_adaln_skips_silu_and_preserves_fp32_coordinates():
    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None

        def forward(self, value):
            self.input = value
            return value, None

    proj = MiniMaxH3AdalnProj.__new__(MiniMaxH3AdalnProj)
    torch.nn.Module.__init__(proj)
    proj.linear = FakeLinear()
    proj.modality_num = 1
    proj.expand_ratio = 1
    proj.hidden_size = 2
    proj.apply_silu = False
    coordinates = torch.tensor([[1.0, -2.0]], dtype=torch.float16)

    output = proj(coordinates)

    torch.testing.assert_close(proj.linear.input, coordinates.to(torch.float32))
    assert output[0].dtype == torch.float32


def test_curve_architecture_fields_are_opt_in():
    full = MiniMaxH3DiTArchConfig.from_mapping({})
    curve = MiniMaxH3DiTArchConfig.from_mapping(
        {"adaln_curve_grid": 1025, "time_embed_dim": 8}
    )

    assert full.adaln_curve_grid is None
    assert full.time_embed_dim == 2688
    assert curve.adaln_curve_grid == 1025
    assert curve.time_embed_dim == 8
    assert curve.adaln_curve_base_dim == 8


def test_curve_extension_pads_base_table_and_projection_weights() -> None:
    arch = MiniMaxH3DiTArchConfig.from_mapping(
        {
            "adaln_curve_grid": 5,
            "adaln_curve_base_dim": 2,
            "time_embed_dim": 3,
        }
    )
    table = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    projection = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    padded_table = _prepare_h3_checkpoint_weight(
        "adaln_t_table", table, arch=arch
    )
    padded_projection = _prepare_h3_checkpoint_weight(
        "blocks.0.adaln_proj.linear.weight", projection, arch=arch
    )

    assert padded_table.shape == (5, 3)
    assert padded_projection.shape == (4, 3)
    torch.testing.assert_close(padded_table[:, :2], table)
    torch.testing.assert_close(padded_projection[:, :2], projection)
    assert torch.count_nonzero(padded_table[:, 2]) == 0
    assert torch.count_nonzero(padded_projection[:, 2]) == 0


def test_curve_extension_prefers_full_overlay_table_over_padded_base() -> None:
    table = torch.nn.Buffer(torch.empty(5, 3), persistent=True)

    class FakeCurveModel:
        arch = MiniMaxH3DiTArchConfig.from_mapping(
            {
                "adaln_curve_grid": 5,
                "adaln_curve_base_dim": 2,
                "time_embed_dim": 3,
            }
        )

        @staticmethod
        def named_parameters():
            return []

        @staticmethod
        def named_buffers():
            return [("adaln_t_table", table)]

    overlay = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    base = torch.full((5, 2), -1.0)
    loaded = MiniMaxH3DiTModel.load_weights(
        FakeCurveModel(),
        [("adaln_t_table", overlay), ("adaln_t_table", base)],
    )

    assert loaded == {"adaln_t_table"}
    torch.testing.assert_close(table, overlay)


def test_curve_runtime_exposes_logical_qkv_lora_targets():
    assert MiniMaxH3DiTModel.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    assert MiniMaxH3DiTModel.stacked_params_mapping == [
        (".attn.qkv_proj", ".attn.q_proj", "q"),
        (".attn.qkv_proj", ".attn.k_proj", "k"),
        (".attn.qkv_proj", ".attn.v_proj", "v"),
    ]
