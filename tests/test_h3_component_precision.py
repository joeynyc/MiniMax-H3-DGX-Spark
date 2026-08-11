# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch


def test_production_quantization_routes_fp8_and_nf4_by_component() -> None:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_component_quant_config,
    )
    from vllm_omni.quantization import build_quant_config

    payload = json.loads(Path("/etc/minimax-h3/fp8-quant.json").read_text())
    config = build_quant_config(payload)
    transformer = _resolve_component_quant_config(config, "transformer")
    text_encoder = _resolve_component_quant_config(config, "text_encoder")

    assert transformer.get_name() == "fp8"
    assert transformer.activation_scheme == "dynamic"
    assert text_encoder.get_name() == "bitsandbytes"
    assert text_encoder.quant_type == "nf4"
    assert text_encoder.compress_statistics is True


def test_text_encoder_precision_contract_is_nf4_only() -> None:
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        _resolve_nf4_settings,
    )

    enabled, compressed = _resolve_nf4_settings(
        SimpleNamespace(
            get_name=lambda: "bitsandbytes",
            quant_type="nf4",
            compress_statistics=True,
            ignored_layers=[],
        )
    )
    assert (enabled, compressed) == (True, True)


def test_video_vae_is_fp16_while_audio_vae_remains_fp32(monkeypatch) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeRemote(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
            "sample_rate": 32000,
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cpu"),
    )
    audio_vae = vae_module.MiniMaxH3AudioVAE(
        "unused",
        device=torch.device("cpu"),
    )

    assert next(video_vae.parameters()).dtype == torch.float16
    assert next(audio_vae.parameters()).dtype == torch.float32


def test_video_vae_reference_encoders_autocast_fp32_inputs_to_fp16() -> None:
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        MiniMaxH3VideoVAE,
    )

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 1, kernel_size=1).half()
            self.parallel_tiling = True

        def _encode_fp32_input(self) -> list[torch.Tensor]:
            value = torch.ones(1, 1, 1, 2, 2, dtype=torch.float32)
            return [self.conv(value)]

        def encode_images(
            self,
            _image: object,
            *,
            use_fp16_latent: bool,
        ) -> list[torch.Tensor]:
            assert use_fp16_latent
            return self._encode_fp32_input()

        def encode_videos(
            self,
            _frames: object,
            *,
            use_fp16_latent: bool,
        ) -> list[torch.Tensor]:
            assert use_fp16_latent
            return self._encode_fp32_input()

    video_vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(video_vae)
    video_vae.model = FakeModel()
    video_vae.config_dict = {
        "latent_channels": 1,
        "latents_mean": [0.0],
        "latents_std": [1.0],
    }

    image_rows = video_vae.encode_image(object())
    video_rows, shape = video_vae.encode_video("frames")

    assert image_rows.dtype == torch.float32
    assert video_rows.dtype == torch.float32
    assert shape == (1, 2, 2)
