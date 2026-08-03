# Measured acceptance result

These results describe one FL2VA text-to-audio-video request on one NVIDIA DGX
Spark. They are not a quality study, vendor benchmark, or cross-machine claim.

## Runtime identity

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, SM121 |
| Host architecture | ARM64 (`aarch64`) |
| GPU count | 1 |
| Base image | `vllm/vllm-omni:minimax-h3` |
| Base-image digest | `sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e` |
| vLLM-Omni | `0.1.dev2381+g310b4b477` |
| vLLM | `0.26.0` |
| Quantization | Online dynamic FP8 |
| Attention | PyTorch SDPA |
| Execution | Eager |

The image emitted a vLLM/vLLM-Omni major/minor mismatch warning. It is recorded
here rather than hidden; this exact pinned combination passed the measured
request.

## Load and memory

| Measurement | Result |
|---|---:|
| Checkpoint shards | 13/13 loaded |
| Shard loading | 154.75 seconds |
| Total model loading | 532.257 seconds |
| Reported model memory | 89.1659 GiB |
| Post-request worker GPU allocation | 92,957 MiB |
| Post-request host memory available | 21 GiB |
| Post-request swap used | 2.8 GiB of 15 GiB |

## Acceptance request

```text
task: t2va
prompt: Macro soldering a PCB under warm bench light, soft room tone.
width: 768
height: 448
fps: 24
duration: 2.0 seconds
num_inference_steps: 20
flow_shift: 12
audio_flow_shift: 3.0
seed: 42
```

| Measurement | Result |
|---|---:|
| HTTP status | 200 |
| Server end-to-end latency | 154.956 seconds |
| Mean denoising-step latency | 7.748 seconds |
| File size | 698,852 bytes |
| Video | H.264 Constrained Baseline, 768x448, yuv420p |
| Frames | 56 at constant 24 fps |
| Video duration | 2.333 seconds |
| Audio | AAC-LC, stereo, 32 kHz |
| Audio duration | 2.357 seconds |
| Audio level | -42.4 dB mean, -29.0 dB peak |
| Full FFmpeg decode | Passed |
| Midpoint inspection | Coherent macro PCB/soldering subject and warm lighting |

The generated artifact is not distributed in this public repository. See
[the model-license notice](../MODEL-LICENSE.md).

## Patch tests

Five focused tests passed inside the exact base image:

1. grouped QKV layout normalization;
2. native parameter-loader contract preservation;
3. fused fc1 layout validation;
4. native CUDA FP8 activation-quantizer binding; and
5. BF16 AdaLN activation input after FP8 weight conversion.
