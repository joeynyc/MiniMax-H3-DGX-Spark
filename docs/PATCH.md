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
| Compiled FP8 | Triton could not lower the FP8 scalar used by the compiled activation quantizer on SM121 | Bind the already-supported native CUDA op |
| Native FP8 | AdaLN converted its activation to the now-FP8 weight dtype before activation quantization | Keep AdaLN input BF16 |
| FlashAttention-4 | CuTe variable-length attention failed to compile for H3's packed sequence shape | Select PyTorch SDPA |

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

- `--enforce-eager`: avoids the regional/QuantFP8 compile path that failed on
  SM121.
- `--diffusion-attention-backend TORCH_SDPA`: avoids the failing FA4 CuTe
  variable-length kernel.
- `--force-cutlass-fp8`: retains the working CUTLASS FP8 linear path.
- six ignored projection/output layers: preserves the model's existing
  sensitive-layer precision contract.

Every item is narrow by design. Any upstream-image or backend change should be
treated as a new configuration and revalidated from cold start through decoded
audio-video output.
