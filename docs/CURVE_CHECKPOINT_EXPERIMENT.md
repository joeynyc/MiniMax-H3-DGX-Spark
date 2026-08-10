# AdaLN curve-checkpoint experiment

This began as a one-Spark A/B and became the production checkpoint decision.
On 2026-08-08, the FL2VA transformer was converted to the same AdaLN curve form
used by the ComfyUI pruned checkpoint while retaining the original BF16
weights for the rest of the transformer. All 50 transformer blocks remain;
this is curve compression, not layer deletion.

The derived checkpoint was built locally from the full FL2VA model and the
curve tensors in
`minimax_h3_fl2va_pruned_int8_convrot.safetensors`. The converter does not
redistribute weights. It replaces the 51 timestep-conditioned AdaLN
projections, removes the original timestep embedder, adds the 1,025 by 8 curve
table, and links the unchanged text encoder, tokenizer, processor, and VAEs
from the base model.

## Matched request

Both variants used the CUTLASS online-FP8, cuDNN-attention, regional-compile,
no-cache profile and this request:

```text
task: t2va
prompt: Macro soldering a PCB under warm bench light, soft room tone.
width: 768
height: 448
fps: 24
duration requested: 4 seconds
num_inference_steps: 20
seed: 42
audio: enabled
```

## Results

| Measurement | Full FL2VA | Curve FL2VA | Difference |
|---|---:|---:|---:|
| Transformer files | 66,280,485,936 B | 40,225,723,352 B | -39.31% |
| Online-FP8 linears bound | 260 | 209 | -51 AdaLN projections |
| Container start to healthy | 592 s | 541 s | -8.61% |
| Transformer shard load | 140.94 s | 153.56 s | +8.95% |
| Minimum available memory during startup | 13.1 GiB | 34.4 GiB | +21.3 GiB |
| Idle available memory | about 22 GiB | 34.4 GiB | about +12 GiB |
| Client render time | 233.548 s | 230.875 s | -1.14% |
| Minimum available memory during render | 20.9 GiB | 33.0 GiB | +12.1 GiB |
| Output | 107 frames / 4.482 s | 107 frames / 4.482 s | equal shape |

The curve checkpoint was cold in the filesystem cache, so its slower shard
load is not evidence that curve loading is intrinsically slower. Total
readiness improved, but a repeated cache-controlled startup pair is required
before making a general startup-speed claim.

Frame-aligned comparison measured SSIM 0.702122 and average video PSNR
18.843723 dB. Those values demonstrate that the same seed does not preserve an
identical trajectory; they do not by themselves measure perceptual quality.
Start, midpoint, end, and a 22-frame sheet were inspected for each output.
Both kept the board, soldering tool, hand, warm lighting, and continuous action
coherent. Neither showed an obvious decode failure, structural collapse, or
temporal corruption. The curve result changed camera composition and tool
motion and appeared slightly softer in parts of the sequence.

The generated audio also changed. The full output measured -22.9 dB mean and
-9.1 dB maximum; the curve output measured -27.8 dB mean and -13.6 dB
maximum. The decoded waveform residual measured -19.25 dB RMS overall. The
FFmpeg `apsnr` summary was implausibly high and contradicted both decoded
loudness and direct residual measurement, so it is not used as acceptance
evidence for this pair. `scripts/compare-quality.sh` now always records the
residual waveform alongside optional APSNR output.

## Decision

The experiment establishes a substantial memory benefit and no meaningful
throughput benefit for this workload. Memory headroom is a product requirement:
Tradecraft must be able to retain task-specific image models and small language
models beside H3. The full checkpoint therefore does not satisfy the production
architecture even though it remains a useful offline comparison source.

The implementation is deliberately gated by `adaln_curve_grid` in the derived
transformer config. Ordinary MiniMax H3 checkpoints continue through the
existing full AdaLN code path for offline tests, but the production launcher
and preflight refuse to serve a canonical `FL2VA` or `Ref2VA` partition without
valid curve metadata. The full sources are retained under `FL2VA-Full` and
`Ref2VA-Full`; there is no production full/curve selector.

## Canonical Ref2VA acceptance

The matching Ref2VA curve checkpoint was then built and installed as the
canonical `Ref2VA` partition. It passed all 13 shard and 532 tensor index
checks, contained 102 curve AdaLN projection tensors plus the 1,025 by 8 table,
and contained no original `time_embedder` tensors.

| Measurement | Canonical curve Ref2VA |
|---|---:|
| Model load | 77.1211 GiB |
| FP8 linears bound | 209 |
| Transformer shard load | 152.86 s |
| Startup image-conditioned warmup | 81.225 s |
| Image + voice transport render, 2 steps | 35.967 s |
| Video-reference transport render, 2 steps | 190.354 s |
| Image + voice quality render, 20 steps | 259.666 s |
| Available memory during the 20-step render | about 28 GiB |
| OOM kills / restarts | 0 / 0 |

All three outputs passed full FFmpeg decode as 107-frame, 768x448 H.264 video
with AAC-LC stereo audio. The two-step requests are transport checks, not
quality evidence. The inspected 20-step frame sheet retained the two faces,
elongated necks, microphones, mixing desk, split-color lighting, and continuous
performance without visible structural or temporal collapse. Its audio was
non-silent at -19.5 dB mean and -2.5 dB maximum.

The first video-reference attempt used an existing 25-second catalog asset and
was correctly rejected by H3's 2-15 second reference-duration contract. A
four-second FFmpeg-derived segment passed. This was input validation behavior,
not a curve-checkpoint failure.

Both `FL2VA` and `Ref2VA` now name curve-compressed production checkpoints.
Their full sources remain under the `*-Full` names for conversion, regression
comparison, and upstream research only.

## Auxiliary-model co-residency

The product requirement is not merely a smaller H3 number; another useful
model must remain operable on the device. With canonical curve Ref2VA resident,
Ollama loaded the official `qwen3.5:9b` tag as a 5.6 GB, 100%-GPU VLM with an
explicit 8,192-token context. While a Ref2VA image-conditioned request was
active, it correctly inspected an eight-panel contact sheet and identified its
two dominant hair colors in 12.411 seconds. The concurrent H3 request completed
in 36.670 seconds and its 107-frame, 4.482-second H.264/AAC output passed full
decode.

Both GPU processes remained resident, H3 stayed healthy, and the host retained
about 22 GiB available memory with zero OOM kills or container restarts. This
directly validates the architectural objective: compressed H3 can coexist with
and make progress beside the small labeling/analysis model class Tradecraft
needs. The tested Ollama artifact occupies 6.6 GB on disk and is available under
the Apache-2.0 license. Packaging should pin and pull this official tag during
optional local-VLM setup rather than bundle its weights or rely on an altered
community derivative.

## Tutu 8-NFE curve extension

The upstream Tutu step100 LoRA modifies the full timestep embedder as well as
51 AdaLN projections, so it cannot be applied directly to the eight-dimensional
production curve. Dropping those tensors was rejected. A joint residual SVD
over the timestep delta and every AdaLN LoRA input found that one additional
curve direction was sufficient: the worst relative projection error fell from
0.981517 in the original basis to 0.000727 in the nine-dimensional basis.

`scripts/prepare-h3-curve-tutu-adapter.py` now builds an atomic, local-only
bundle containing a tiny curve-table overlay, symlinks to the unchanged model
shards, and a 368-module residual adapter. The loader zero-extends the original
eight-wide curve weights and prefers the full-width overlay table regardless of
shard enumeration order. The bundle remains checksum-pinned, model-derived,
and unqualified until hardware and human gates are recorded.

On one NVIDIA GB10, the extended bundle loaded at 38.5925 GiB and the worker
reported 38.94 GiB after initialization. The fixed-seed 768x448 hands/tools case
produced valid 107-frame H.264 video plus stereo AAC audio in 141.720 seconds on
the first request and 129.503 seconds warm. The matched naked eight-step control
took 111.884 seconds; the warmed 20-step quality reference took 225.807 seconds.
The two Tutu runs had identical decoded video-frame hashes. Their generated
audio waveforms differed, although both measured -27.4 dB mean; the naked
control measured -46.9 dB mean.

The inspected Tutu frame sheet retained the watch mechanism, dial, hands, warm
bench lighting, and stable macro composition without an obvious decode or
temporal failure. This closes the curve-compatibility and single-case transport
gate, not the complete quality gate. Dialogue synchronization, fast motion,
occlusion, complex camera movement, audio repeatability, and the full human
rubric remain open. The standard eight-dimensional curve checkpoint stays the
production default.
