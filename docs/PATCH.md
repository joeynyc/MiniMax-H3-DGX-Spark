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

## Deliberate runtime settings

- `--diffusion-attention-backend CUDNN_ATTN`: the fastest stable full-compute
  attention backend in the one-Spark sweep; SDPA remains an accepted baseline.
- regional compile: independently completed cold start and decoded requests
  after the native-CUDA FP8 quantizer correction; 52 repeated H3 blocks were
  compiled.
- `--force-cutlass-fp8`: retains the working CUTLASS FP8 linear path.
- six ignored projection/output layers: preserves the model's existing
  sensitive-layer precision contract.

The optional Cache-DiT profile is not part of the compatibility patch. It is a
runtime acceleration with approximate output reuse and is disabled by default.
Its accepted `0.10` configuration and same-seed quality measurements are in
[RESULTS.md](RESULTS.md).

Every item is narrow by design. Any upstream-image, attention backend,
execution mode, cache configuration, or patch change should be treated as a
new configuration and revalidated from cold start through decoded audio-video
output.
