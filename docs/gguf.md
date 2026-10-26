# What the GGUF patches do

`setup.sh` applies them after the MXFP4 recipe's patches ([patches.md](patches.md)):

| file | applies to |
|---|---|
| `patches/gguf/sglang-gguf.patch` | sgl-project/sglang `65633129` with tiers 0 to 2 applied |
| `patches/gguf/kt-kernel-gguf.patch` | kvcache-ai/ktransformers `95009ea` with `kt-kernel-tree.patch` applied |
| `patches/gguf/kt-llamafile-iq2xs.patch` | the same tree's `third_party/llamafile` |

A switch marked "on" is on in `serve.sh`.

## Weights

- **Dense layers** (`kt_gguf_dense_v41.py`, `quantization/gguf.py`, the loader): every dense tensor is
  read from the GGUF shards and kept in its GGUF type on the GPU (Q5_K and others in this pack). The
  compressor and the indexer are dequantized to bf16 at load (`modules_to_not_convert`). Decode
  multiplies the GGUF rows with vector kernels taken from llama.cpp (`kt_gguf_moe_multi/`); a prefill
  of 64 rows or more dequantizes each weight to bf16 once and runs a bf16 GEMM.
- **Routed experts**: kt-kernel's LLAMAFILE backend computes them from the GGUF's rows
  (`--kt-method LLAMAFILE`, `KT_GGUF_ZEROCOPY=1`, on): no arena copy of the experts at load.
- **Engram**: the two tables are read row by row from the Engram GGUF (IQ4_NL, 144 bytes a row)
  and dequantized on the GPU (`engram-adapter/`).

## The expert cache (kt-kernel)

`KT_GGUF_EXPERT_CACHE_GIB` (sized by `serve.sh` from the node's memory) is kt-kernel's own RAM tier
for the experts: an expert's gate, up and down rows are read once from the file with `O_DIRECT`
into fixed slots on the node and computed from there; when the cache is full, the least recently
used experts give their slots back. Reader threads on the node's last cores
(`KT_GGUF_PREFETCH_CPUS`, `KT_GGUF_PREFETCH_THREADS`) keep many reads in flight. The reads bypass
the page cache (plain reads only where the file system refuses `O_DIRECT`), so the cache's size is
the RAM the experts take.

`KT_GGUF_CACHE_IQ1M_T=1` (on when the node holds every expert and the copy) keeps, beside each
IQ1_M tensor's raw rows, a copy laid out for the CPU decode: 76 bytes a block of 256 values (IQ1_M:
56). Its GEMM decodes 32 values with a shift, a mask and a byte shuffle and then does the same
integer dot products in the same order as iqk's IQ1_M kernel, so the result is bit-identical. An
IQ1_M layer's CPU work in decode goes from 0.94 to 0.55 ms. The raw rows stay for the GPU readers
below.

`kt-llamafile-iq2xs.patch` adds IQ2_XS to the x86 kernels kt-kernel uses (iqk's matrix
multiplication, from ik_llama.cpp); the pack's IQ2_XS tensors run there.

## Hot experts on the GPU

The MXFP4 recipe's hot lend ([tier1.md](tier1.md#16-hot-lend)) with GGUF slots (`KT_HOT_BACKEND=gguf`,
`kt_gguf_hot.py`): one expert a layer is resident, and the KV pool's free pages and the driver's
idle memory hold as many more as fit, in their GGUF bytes. Decode runs them with the vector
kernels above and passes the routes they do not hold to kt-kernel (`-1`).

Every 8 forwards each layer swaps slots whose experts are routed less for experts routed more
(`KT_KV_HOT_DYNAMIC_SLOTS=1`, on). A swap's rows come from the expert cache, or from the file when
the cache does not hold them. They are read on two background threads into a pinned staging pool
(`KT_KV_HOT_DYN_ASYNC=1`, on; 24 buffers of 9.8 MiB), copied to the slots on a side stream at the
next round, and mapped at the round after: the forward never waits on a read. The table update
goes through pinned memory without a host sync. `KT_KV_HOT_DYN_VERIFY=1` reads every swapped slot
back and compares it with the file.

## Prefill

The MXFP4 recipe's streamed prefill, fed from the GGUF (`kt_stream_prefill_gguf_nvme.py`,
`KT_GGUF_STREAM_SOURCE=nvme`, on above 1024 rows): reader threads (`KT_GGUF_STREAM_READERS`, 16)
read each layer's experts with `O_DIRECT` into a ring of pinned buffers (`KT_GGUF_STREAM_RING`, 16),
taking the ones the expert cache holds from RAM, and the GPU multiplies them in their GGUF types
with llama.cpp's MMQ kernels (`kt_gguf_moe_mmq/`, `KT_GGUF_STREAM_MMQ=1`, on), built for the GPU's
arch. The lends work as in the MXFP4 recipe; the lend's copy of the dense weights is read back from
the GGUF instead of kept in host memory (`KT_PREFILL_LEND_MIRROR=gguf`, on).

## Not carried over from the MXFP4 recipe

- Tensor parallelism: the GGUF blocks of an expert do not split across ranks.
- The placement file and the static hot experts: the dynamic slots find the hot experts themselves.
- `wo_a` in FP8: it is a GGUF tensor here.
- The vision tower: the GGUF has none.
