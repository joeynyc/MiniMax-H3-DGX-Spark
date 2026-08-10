# Why the SM121 compatibility patch exists

MiniMax H3 support landed while this bring-up was underway. The pinned image
could recognize the FL2VA pipeline, but the normal single-GPU choices failed at
different layers of the stack on GB10.

## Failure chain

| Attempt | Observed failure | Decision |
|---|---|---|
| BF16 with offload | Unified-memory use approached 118–119 GiB, leaving no practical request headroom | Rejected for one Spark |
| Online INT8 | Request reached `dispatch_scaled_mm`, which reported INT8 unsupported on SM121 | Rejected |
| First online FP8 | Custom H3 loader accepted two arguments while FP8 replayed a native three-argument loader | Repair loader contract |
| Initial compiled FP8 | Triton could not lower the FP8 scalar used by the compiled activation quantizer on SM121 | Bind the already-supported native CUDA op, then independently re-test regional compile |
| Native FP8 | AdaLN converted its activation to the now-FP8 weight dtype before activation quantization | Keep AdaLN input BF16 |
| FlashAttention-4 | CuTe variable-length attention failed to compile for H3's packed sequence shape | Reject FA4; independently test SDPA and cuDNN |
| First Humming load | GB10 does not expose a usable NVML maximum memory clock or bus width | Fall back to GB10 CUDA device properties only |
| Humming weight conversion | Online FP8 left a transposed `[K, N]` view while Humming requires contiguous `[N, K]` | Restore and validate the canonical layout |
| Humming activation conversion | The adapter omitted the online activation schema and silently built a BF16-input kernel | Pass the dynamic per-token FP8 schema explicitly |
| Humming fixed-seed repeat | Stream-K changed the large QKV reduction order between launches | Select Humming's batch-invariant scheduler without enabling vLLM's global BF16 fallback |

## Loader correction

The initial implementation installed custom closures directly on QKV and fc1
parameters. Online FP8 temporarily wraps and later replays native parameter
loaders, including their complete signatures. A two-argument closure therefore
failed before checkpoint shards could load.

The patch instead normalizes grouped checkpoint QKV rows in
`MiniMaxH3DiTModel.load_weights`, immediately before handing the tensor to the
untouched native loader. The fused fc1 loader is also left native; the patch
only validates that its checkpoint row count can split evenly into gate and up
halves.

## FP8 activation correction

The pinned runtime's independently compiled QuantFP8 wrapper reaches a Triton
SM121 code-generation limitation. Its native CUDA `scaled_fp8_quant` operation
does work. During model construction the compatibility module discovers the
online-FP8 linears and binds their activation quantizers to `forward_cuda`.
The measured model bound 260 quantizers.

AdaLN previously cast its input to `self.linear.weight.dtype`. That works while
the weight remains BF16, but online quantization replaces it with an FP8
parameter. The corrected path explicitly keeps the AdaLN activation BF16 and
lets the linear quantization method handle conversion.

## Production AdaLN curve checkpoint

The production model replaces the original timestep MLP and 51 large AdaLN
projections with the checkpoint's 1,025-point, 8-dimensional AdaLN curve and
compact projections. All 50 transformer blocks remain. The runtime performs
the curve interpolation and compact projections in FP32; checkpoints without
`adaln_curve_grid` retain the original full AdaLN implementation for offline
comparison.

`scripts/prepare-h3-curve-checkpoint.py` combines those curve tensors with the
unchanged tensors from an authorized full checkpoint and links the unchanged
text encoder, tokenizer, processor, and VAEs. It never writes into the source
checkpoint and refuses to overwrite an existing output directory.

Production is intentionally stricter than the model implementation. The
launcher and preflight require valid curve metadata in the canonical `FL2VA`
or `Ref2VA` partition and fail before model loading otherwise. Full checkpoints
may be retained as `FL2VA-Full` and `Ref2VA-Full`, but are not selectable
production profiles. The measured rationale and acceptance evidence are in
[CURVE_CHECKPOINT_EXPERIMENT.md](CURVE_CHECKPOINT_EXPERIMENT.md).

An experimental Tutu adapter bundle may declare `adaln_curve_base_dim` smaller
than `time_embed_dim`. The loader pads the base curve table and compact AdaLN
weights with zero columns, then accepts a full-width overlay table. This keeps
the unchanged production shards reusable while allowing a checksum-pinned
adapter-specific curve extension. The ordinary curve path is unchanged when
the two dimensions are equal.

## H3 inference optimizations

The image also carries four behavior-preserving H3 changes developed against
upstream vLLM-Omni and adapted to this older pinned runtime:

1. build each packed attention mask once per request instead of once per
   transformer layer;
2. use vLLM's fused `SiluAndMul` implementation for SwiGLU;
3. refine prompt embeddings and compute RoPE once before the denoise loop; and
4. use direct indexed assignment for modality embeddings whose packed rows are
   disjoint.

The prompt/RoPE change touches the pinned denoise loop and pipeline as well as
the transformer. `minimax_h3_static_conditioning.patch` applies those two
narrow call-site changes during the image build. The transformer retains the
raw-conditioning fallback for direct callers.

Ref2VA also accepts the optional request field
`reference_image_pixel_budget_percent`. Values from 50 through 400 resize each
reference image to approximately that percentage of the selected output pixel
area while preserving its aspect ratio and H3's 32-pixel alignment. Omitting
the field retains the upstream 2,048-pixel short-edge behavior. This control is
opt-in at the API layer. The matched one-Spark experiment in
[RESULTS.md](RESULTS.md) found 100% visually acceptable on its stress fixture
and 2.54x the legacy generation rate, so the companion Studio product selects
100% for ordinary references and reserves 200% for small or numerous details.
That product policy is deliberately separate from the compatibility patch's
backward-compatible default.

These changes preserve the Spark-specific native-CUDA FP8 binding. They do not
enable compilation, replace SDPA, change inference steps, or alter precision.
The fixed-seed acceptance comparison is recorded in
[RESULTS.md](RESULTS.md); normal run-to-run system variance means its timing is
evidence for a useful improvement, not a universal throughput claim.

## Humming FP8 correction

The installed Humming kernel can consume H3's existing online-FP8 contract:
dynamic per-token E4M3 activations, static per-tensor E4M3 weights, BF16 output,
and FP32 accumulation. Three integration details prevented it from being a
safe drop-in replacement.

First, Humming's bandwidth heuristic queried NVML fields that GB10 does not
currently expose. The compatibility patch falls back to CUDA's reported
8,533,000 kHz memory clock and 256-bit bus only when the device identifies as
GB10; unknown devices still fail rather than receiving a guessed value.
Second, online FP8 stores the CUTLASS-friendly transposed weight view, so the
adapter restores contiguous `[N, K]` storage and rejects any other unexpected
shape or stride. Third, the original adapter did not pass its activation
quantization key to `prepare_humming_layer`, producing a BF16-input kernel.
The corrected adapter explicitly translates the dynamic per-token key into
Humming's input schema.

Humming's default Stream-K schedule was not fixed-seed repeatable for H3's
large QKV projection on GB10. `VLLM_HUMMING_BATCH_INVARIANT=1` disables that
schedule inside Humming only. The launcher deliberately does not set vLLM's
global `VLLM_BATCH_INVARIANT`, because the online-FP8 method interprets that
global setting by dequantizing non-CUTLASS weights and running BF16 linear
operations instead.

## Production-shape startup warmup

The generic vLLM-Omni diffusion warmup used one denoise step at 512x512 and
added synthetic input conditions for every modality the model advertises.
That is not a valid FL2VA text-to-video request: a one-entry sigma schedule is
invalid, and FL2VA rejects the generic audio condition.

The model now declares a two-step 768x448 warmup using a four-second request
(96 requested frames before H3's native temporal alignment). The shared
diffusion engine reads optional model-class warmup attributes, while retaining
its existing defaults for every other pipeline. Startup fails closed if this
real generation path fails, and Compose does not report the service healthy
until the API becomes available after warmup.

Humming, Triton, and TorchInductor caches share a dedicated persistent Compose
volume. Recreating the container therefore reuses kernels compiled by the
startup warmup. The volume is tied to the pinned image and trusted runtime; it
is not a portable artifact for arbitrary toolchain changes.

## Deliberate runtime settings

- `--diffusion-attention-backend CUDNN_ATTN`: the fastest stable full-compute
  attention backend in the one-Spark sweep; SDPA remains an accepted baseline.
- regional compile: independently completed cold start and decoded requests
  after the native-CUDA FP8 quantizer correction; 52 repeated H3 blocks were
  compiled.
- `--force-cutlass-fp8`: retains the fastest validated full-compute linear
  path and is the default.
- `H3_LINEAR_BACKEND=HUMMING`: selects the corrected deterministic Humming
  path only for the historical eager/no-cache profile. The launcher rejects
  combinations with regional compile or Cache-DiT because they were not
  validated together.
- `H3_STARTUP_WARMUP=true`: gates readiness on a two-step production-shape
  generation and seeds the persistent kernel caches. `false` is a diagnostic
  rollback to the upstream lazy-first-request behavior.
- six ignored projection/output layers: preserves the model's existing
  sensitive-layer precision contract.

The optional Cache-DiT profile is not part of the compatibility patch. It is a
runtime acceleration with approximate output reuse and is disabled by default.
Its accepted `0.10` configuration and same-seed quality measurements are in
[RESULTS.md](RESULTS.md).

Every item is narrow by design. Any upstream-image, attention backend,
execution mode, cache configuration, linear backend, or patch change should
be treated as a new configuration and revalidated from cold start through
decoded audio-video output.
