import inspect

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
)


def test_minimax_h3_warmup_uses_production_shape() -> None:
    assert MiniMaxH3Pipeline.dummy_run_num_frames == 56
    assert MiniMaxH3Pipeline.dummy_run_num_inference_steps == 2
    assert MiniMaxH3Pipeline.dummy_run_height == 448
    assert MiniMaxH3Pipeline.dummy_run_width == 768
    assert MiniMaxH3Pipeline.dummy_run_use_image_condition is False
    assert MiniMaxH3Pipeline.dummy_run_use_audio_condition is False
    assert MiniMaxH3Pipeline.dummy_run_prompt


def test_diffusion_warmup_reads_model_specific_shape_before_readiness() -> None:
    source = inspect.getsource(DiffusionEngine._dummy_run)

    assert 'model_warmup_attr("dummy_run_num_inference_steps", 1)' in source
    assert 'model_warmup_attr("dummy_run_height", 512)' in source
    assert 'model_warmup_attr("dummy_run_width", 512)' in source
    assert 'model_warmup_attr("dummy_run_use_image_condition", supports_image_input)' in source
    assert 'model_warmup_attr("dummy_run_use_audio_condition", supports_audio_input)' in source
    assert "if use_image_condition:" in source
    assert "if use_audio_condition:" in source
    assert "dummy model warmup completed" in source
