# What the GGUF patches do

`setup.sh` applies them after the MXFP4 recipe's patches ([patches.md](patches.md)):

| file | applies to |
|---|---|
| `patches/gguf/sglang-gguf.patch` | sgl-project/sglang `66f79f1a` with tiers 0 to 2 applied |
| `patches/gguf/kt-kernel-gguf.patch` | kvcache-ai/ktransformers `95009ea` with `kt-kernel-tree.patch` applied |
| `patches/gguf/kt-llamafile-iq2xs.patch` | the same tree's `third_party/llamafile` |

A switch marked "on" is on in `serve.sh`.

## Weights

- **Dense layers** (`kt_gguf_dense_v41.py`, `quantization/gguf.py`, the loader): every dense tensor is
  read from the GGUF shards and kept in its GGUF type on the GPU (Q5_K and others in this pack). The
  compressor and the indexer are dequantized to bf16 at load (`modules_to_not_convert`). Decode
  multiplies the GGUF rows with vector kernels taken from llama.cpp (`kt_gguf_moe_multi/`); a prefill
  of 64 rows or more dequantizes each weight to bf16 once and runs a bf16 GEMM.
- **Embedding**: the token embedding stays in its GGUF type (Q5_K, 434 MiB instead of a 1,262 MiB bf16
  copy); the rows a batch looks up are opened in float32 in gguf-py's order of operations, so the
  embeddings are the same values the bf16 copy held.
- **RoPE tables**: built for the served context (`SGLANG_CONTEXT_LENGTH`) instead of the model's
  1,048,576 positions (two tables of 512 MiB each at 1M; their rows depend only on the position).
- **Routed experts**: kt-kernel's LLAMAFILE backend computes them from the GGUF's rows
  (`--kt-method LLAMAFILE`, `KT_GGUF_ZEROCOPY=1`, on): no arena copy of the experts at load.
- **Engram**: the two tables are read row by row from the Engram GGUF (IQ4_NL, 144 bytes a row)
  and dequantized on the GPU (`engram-adapter/`).

## The expert cache (kt-kernel)

`KT_GGUF_EXPERT_CACHE_GIB` (sized by `serve.sh` from the node's memory) is kt-kernel's own RAM tier
for the experts: an expert's gate, up and down rows are read once from the file with `O_DIRECT`
into fixed slots on the node and computed from there; when the cache is full, the least recently
used experts give their slots back. Reader threads keep many reads in flight
([Drive reads](#drive-reads) says where they run and how many there are). The reads bypass
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

## Drive reads

Four kinds of threads read the GGUF from the drive: the expert cache's readers, the hot lend's
swaps, the streamed prefill's readers and the engram reads. `serve.sh` keeps the first three off
the CPUs the experts compute on, and sizes them from the drive.

- **How many readers.** `setup.sh` runs `tools/io-probe.py` once on the GGUF's drive: random 512 KiB
  and 2 MiB `O_DIRECT` reads from 1 to 32 threads. It writes `state/gguf-<name>/io-probe.json`.
  The cache and the prefill each get the fewest readers that reach 90% of the drive's best rate
  (`KT_GGUF_PREFETCH_THREADS`, `KT_GGUF_STREAM_READERS`).
- **Where, with SMT.** The readers and the swaps run on the second hardware threads of the node's
  last cores, one for every four readers (`KT_IO_CPUS`). The experts compute on every core.
- **Where, without SMT.** The readers and the swaps share the node's last cores, as many as
  reading the drive at full speed keeps busy (the probe's rate times its CPU time a GB, rounded
  up). The experts compute on the rest.
- **The engram reads** are left to the scheduler. `DSV41_ENGRAM_IO_CPUS` pins them; pinned to the
  readers' CPUs or to the other second hardware threads, the decode was a few percent slower on
  some texts.
- **Without a probe** (a setup from before it, or a GGUF moved to another drive), `serve.sh`
  takes 16 readers: four second hardware threads with SMT, one core without. Remove
  `state/gguf-<name>/io-probe.json` and run `GGUF=<dir> ./setup.sh` to measure again.

The launch prints the result, e.g. `drive reads: 16 cache readers and 16 prefill readers on CPUs
92,93,94,95`. Every value can be set by hand: `KT_IO_CPUS`, `DSV41_ENGRAM_IO_CPUS`,
`KT_CPU_THREADS` and the two reader counts.

What it was measured on, two launches each:

- **One EPYC 7452 socket (SMT), an SN5100 (16 readers).** The experts on all 32 cores and the
  readers on four second hardware threads decode 2 to 10% faster than with four whole cores given
  to the readers. That holds at 64 GB and with every expert in RAM. The prefill does not change.
- **An i5-9600K (6 cores, no SMT), a KIOXIA EXCERIA PLUS G2 (8 readers).** One core stays with the
  reads. Computing on all six cores decoded 3 to 6% slower.

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

## Turing (SM75)

`serve.sh` lets an RTX 20 card through and sets it up as below (all of it inert on SM80 and newer):

- **Attention**: SGLang's Triton sparse decode for Ada (SM89, tier 0) serves SM75 too. Triton has no
  FP8 type and no bf16 dot below SM80, so the kernel opens the e4m3 KV bytes from their bit fields
  (the same values as the FP8 conversion: checked for all 256 bytes) and takes its dots in float32.
  The compressed KV's FP4 scale is rounded to e4m3 with integer ops (ties to even, as the cvt does).
- **bf16**: SGLang switches pre-SM80 GPUs to fp16; `KT_SM75_BF16=1` and `--dtype bfloat16` keep bf16,
  which DSV4's JIT kernels need. The MoE router's bf16 GEMM runs in float32.
- **sgl-kernel ops**: the sgl-kernel wheel carries sm_80 and newer code only. `kt_sm75_ops.py` builds
  its GGUF, activation, RoPE, top-k and `moe_sum` sources (in SGLang's tree) for the GPU's arch with
  torch's JIT loader (`setup.sh` does it once) and registers them as those ops' CUDA kernels.
- **Prefill**: the dense layers' prefill runs on MMQ in their GGUF types (`SGLANG_GGUF_DEQUANT_ROWS`,
  `SGLANG_DSV4_WO_A_DEQUANT_ROWS`): the bf16 copy path needs a bf16 cuBLAS GEMM.
- **Context and chunk**: a card under 12 GB defaults to 32,768 tokens. The prefill chunk is 2,048
  tokens with no calibration launch: on SM75 the TileLang mHC kernels (`hc_split_sinkhorn`,
  `mhc_post`) fail to launch inside the server on 8,192-token chunks (`CUDA_ERROR_INVALID_VALUE`;
  they run at that size on their own, the cause is not found), and the CPU sets the prefill's pace.

V100 (SM70) is not covered: CUDA 13 dropped it, so it would take a CUDA 12 build of the whole stack.

## Not carried over from the MXFP4 recipe

- Tensor parallelism: the GGUF blocks of an expert do not split across ranks.
- The placement file and the static hot experts: the dynamic slots find the hot experts themselves.
- `wo_a` in FP8: it is a GGUF tensor here.
- The vision tower: the GGUF has none.
