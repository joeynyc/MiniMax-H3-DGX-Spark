ARG BASE_IMAGE=vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax H3 DGX Spark compatibility image" \
      org.opencontainers.image.description="SM121 online-FP8 compatibility layer for MiniMax H3 FL2VA and Ref2VA" \
      org.opencontainers.image.source="https://github.com/joeynyc/MiniMax-H3-DGX-Spark" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --chmod=0644 patches/minimax_h3_transformer.py \
  /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py
COPY --chmod=0644 patches/minimax_h3_static_conditioning.patch /tmp/minimax_h3_static_conditioning.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3 \
      < /tmp/minimax_h3_static_conditioning.patch \
    && rm /tmp/minimax_h3_static_conditioning.patch
COPY --chmod=0644 patches/minimax_h3_production_warmup.patch /tmp/minimax_h3_production_warmup.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_production_warmup.patch \
    && rm /tmp/minimax_h3_production_warmup.patch
COPY --chmod=0644 patches/minimax_h3_ref2va_warmup_audio.patch /tmp/minimax_h3_ref2va_warmup_audio.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_ref2va_warmup_audio.patch \
    && rm /tmp/minimax_h3_ref2va_warmup_audio.patch
# Backport vLLM-Omni ae7bee0c (#5752) until the pinned H3 image includes the
# official Ref2VA mixed-input contract (visual references with optional audio).
COPY --chmod=0644 patches/minimax_h3_official_input_matrix.patch /tmp/minimax_h3_official_input_matrix.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_official_input_matrix.patch \
    && rm /tmp/minimax_h3_official_input_matrix.patch
COPY --chmod=0644 patches/minimax_h3_reference_pixel_budget.patch /tmp/minimax_h3_reference_pixel_budget.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_reference_pixel_budget.patch \
    && rm /tmp/minimax_h3_reference_pixel_budget.patch
# Backport the measured component-precision profile from vLLM-Omni commits
# 98e93213, 2e453063, and 6527bc46. These are production invariants, not switches:
# native Qwen projections use NF4, video VAE uses FP16, audio VAE stays FP32.
COPY --chmod=0644 patches/minimax_h3_text_encoder_nf4.patch /tmp/minimax_h3_text_encoder_nf4.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_text_encoder_nf4.patch \
    && rm /tmp/minimax_h3_text_encoder_nf4.patch
COPY --chmod=0644 patches/minimax_h3_video_vae_fp16.patch /tmp/minimax_h3_video_vae_fp16.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_video_vae_fp16.patch \
    && rm /tmp/minimax_h3_video_vae_fp16.patch
COPY --chmod=0644 patches/minimax_h3_video_vae_reference_autocast.patch /tmp/minimax_h3_video_vae_reference_autocast.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/minimax_h3_video_vae_reference_autocast.patch \
    && rm /tmp/minimax_h3_video_vae_reference_autocast.patch
COPY --chmod=0644 patches/humming_gb10_bandwidth.patch /tmp/humming_gb10_bandwidth.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/humming_gb10_bandwidth.patch \
    && rm /tmp/humming_gb10_bandwidth.patch
COPY --chmod=0644 patches/humming_online_fp8_layout.patch /tmp/humming_online_fp8_layout.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/humming_online_fp8_layout.patch \
    && rm /tmp/humming_online_fp8_layout.patch
# Backport vLLM-Omni 9e35149a until the pinned H3 image includes the generic
# mixed-dtype diffusion-LoRA input repair used by the Tutu performance recipe.
COPY --chmod=0644 patches/vllm_omni_diffusion_lora_mixed_dtype.patch /tmp/vllm_omni_diffusion_lora_mixed_dtype.patch
RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/vllm_omni_diffusion_lora_mixed_dtype.patch \
    && rm /tmp/vllm_omni_diffusion_lora_mixed_dtype.patch
COPY --chmod=0755 start-fp8.sh /usr/local/bin/start-minimax-h3
COPY --chmod=0644 scripts/security-common.sh /usr/local/lib/minimax-h3/security-common.sh
COPY --chmod=0644 fp8-quant.json /etc/minimax-h3/fp8-quant.json

ENTRYPOINT ["/usr/local/bin/start-minimax-h3"]
