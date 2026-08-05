from __future__ import annotations

import inspect

import torch

from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MINIMAX_H3_STATIC_CONDITIONING_KWARG,
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    MiniMaxH3MLP,
    MiniMaxH3StaticConditioning,
    _build_packed_attention_mask,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
)


def _packed_fixture() -> dict[str, torch.Tensor | int]:
    return {
        "seq_len": 5,
        "img_pos": torch.tensor([2]),
        "audio_pos": torch.tensor([3]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([True]),
        "cu_seqlens": torch.tensor([0, 4, 5], dtype=torch.int32),
        "img_position_ids": torch.zeros((5, 3), dtype=torch.int32),
    }


def test_packed_attention_mask_is_precomputed_from_layout_lengths() -> None:
    assert _build_packed_attention_mask(
        used_seqlen=5,
        packed_seqlen=5,
        device=torch.device("cpu"),
    ) is None
    mask = _build_packed_attention_mask(
        used_seqlen=4,
        packed_seqlen=5,
        device=torch.device("cpu"),
    )
    assert mask is not None
    assert mask.tolist() == [[True, True, True, True, False]]
    assert "attn_mask" in inspect.signature(MiniMaxH3Attention._run_packed_attention).parameters


def test_denoise_branch_accepts_precomputed_static_conditioning() -> None:
    static = MiniMaxH3StaticConditioning(
        refined_prompt_embeds=torch.zeros((2, 4), dtype=torch.bfloat16),
        rope_freqs=torch.zeros((5, 1), dtype=torch.float32),
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=_packed_fixture(),
        text_embeddings=torch.zeros((2, 4)),
        token_tags=torch.tensor([1, 1, 0, 2, -1]),
        device=torch.device("cpu"),
        static_conditioning=static,
    )
    assert branch.static_kwargs[MINIMAX_H3_STATIC_CONDITIONING_KWARG] is static
    assert "prompt_embeds" not in branch.static_kwargs
    assert "img_position_ids" not in branch.static_kwargs
    assert "refiner_packed_seq_params" not in branch.static_kwargs


def test_denoise_branch_preserves_raw_conditioning_fallback() -> None:
    branch = MiniMaxH3DenoiseBranch(
        packed=_packed_fixture(),
        text_embeddings=torch.zeros((2, 4)),
        token_tags=torch.tensor([1, 1, 0, 2, -1]),
        device=torch.device("cpu"),
    )
    assert "prompt_embeds" in branch.static_kwargs
    assert "img_position_ids" in branch.static_kwargs
    assert "refiner_packed_seq_params" in branch.static_kwargs


def test_pipeline_prepares_static_conditioning_before_denoising() -> None:
    source = inspect.getsource(MiniMaxH3Pipeline.diffuse)
    assert "prepare_static_conditioning" in source
    assert "static_conditioning=static_conditioning" in source


def test_transformer_contains_fused_and_direct_embedding_paths() -> None:
    mlp_source = inspect.getsource(MiniMaxH3MLP)
    embed_source = inspect.getsource(MiniMaxH3DiTModel._embed)
    assert "SiluAndMul" in mlp_source
    assert ".chunk(" not in mlp_source
    assert embed_source.count("index_copy_") == 3
    assert "embeddings.index_add_" not in embed_source
