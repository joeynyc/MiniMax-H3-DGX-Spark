ARG BASE_IMAGE=vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax H3 DGX Spark compatibility image" \
      org.opencontainers.image.description="SM121 online-FP8 compatibility layer for MiniMax H3 FL2VA" \
      org.opencontainers.image.source="https://github.com/joeynyc/MiniMax-H3-DGX-Spark" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --chmod=0644 patches/minimax_h3_transformer.py \
  /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py

# Harden the async result pump: a cancelled output future used to kill the sole
# DiffusionResultPump thread, leaving the server healthy-looking but unable to
# return any video. The script fails the build if its anchors no longer match.
COPY --chmod=0755 patches/apply-result-pump-hotfix.py \
  /usr/local/lib/minimax-h3/apply-result-pump-hotfix.py
RUN python3 /usr/local/lib/minimax-h3/apply-result-pump-hotfix.py

COPY --chmod=0755 start-fp8.sh /usr/local/bin/start-minimax-h3
COPY --chmod=0644 scripts/security-common.sh /usr/local/lib/minimax-h3/security-common.sh
COPY --chmod=0644 fp8-quant.json /etc/minimax-h3/fp8-quant.json

ENTRYPOINT ["/usr/local/bin/start-minimax-h3"]
