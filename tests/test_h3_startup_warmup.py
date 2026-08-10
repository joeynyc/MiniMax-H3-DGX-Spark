import inspect
import os
import subprocess
import sys

from PIL import Image
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    _reference_image_shape,
    _validate_ref2va_reference_counts,
)
from vllm_omni.errors import OmniClientError
import pytest


def test_minimax_h3_warmup_uses_production_shape() -> None:
    assert MiniMaxH3Pipeline.dummy_run_num_frames == 96
    assert MiniMaxH3Pipeline.dummy_run_num_inference_steps == 2
    assert MiniMaxH3Pipeline.dummy_run_height == 448
    assert MiniMaxH3Pipeline.dummy_run_width == 768
    assert MiniMaxH3Pipeline.dummy_run_use_image_condition is False
    assert MiniMaxH3Pipeline.dummy_run_use_audio_condition is False
    assert MiniMaxH3Pipeline.dummy_run_audio_as_waveform_tuple is True
    assert MiniMaxH3Pipeline.dummy_run_prompt


def test_diffusion_warmup_reads_model_specific_shape_before_readiness() -> None:
    source = inspect.getsource(DiffusionEngine._dummy_run)

    assert 'model_warmup_attr("dummy_run_num_inference_steps", 1)' in source
    assert 'model_warmup_attr("dummy_run_height", 512)' in source
    assert 'model_warmup_attr("dummy_run_width", 512)' in source
    assert 'model_warmup_attr("dummy_run_use_image_condition", supports_image_input)' in source
    assert 'model_warmup_attr("dummy_run_use_audio_condition", supports_audio_input)' in source
    assert 'model_warmup_attr("dummy_run_audio_as_waveform_tuple", False)' in source
    assert "if use_image_condition:" in source
    assert "if use_audio_condition:" in source
    assert "if audio_as_waveform_tuple else dummy_audio" in source
    assert '"aspect_ratio": "16:9"' in source
    assert "dummy model warmup completed" in source


def test_ref2va_warmup_supplies_required_visual_condition() -> None:
    env = os.environ.copy()
    env["H3_RIG_MODE"] = "ref2va"
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 "
            "import MiniMaxH3Pipeline; "
            "print(MiniMaxH3Pipeline.dummy_run_use_image_condition, "
            "MiniMaxH3Pipeline.dummy_run_use_audio_condition, "
            "MiniMaxH3Pipeline.dummy_run_audio_as_waveform_tuple)",
        ],
        env=env,
        text=True,
    )

    assert output.strip().splitlines()[-1] == "True True True"


def test_ref2va_accepts_visual_references_without_forcing_audio() -> None:
    _validate_ref2va_reference_counts(1, 0, 0)

    with pytest.raises(OmniClientError, match="at least one image or video"):
        _validate_ref2va_reference_counts(0, 0, 1)


def test_ref2va_reference_pixel_budget_is_opt_in() -> None:
    image = Image.new("RGB", (1672, 941))

    assert _reference_image_shape(image) == (3648, 2048)
    assert _reference_image_shape(
        image,
        output_width=768,
        output_height=448,
        pixel_budget_percent=100,
    ) == (768, 448)

    with pytest.raises(OmniClientError, match="between 50 and 400"):
        _reference_image_shape(
            image,
            output_width=768,
            output_height=448,
            pixel_budget_percent=401,
        )
