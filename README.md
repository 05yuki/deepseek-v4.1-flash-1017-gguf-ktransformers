# DeepSeek-V4.1-Flash-1017 GGUF on SGLang + KTransformers (one 16 GB GPU, 64 GB of RAM)

Run [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
with [SGLang](https://github.com/sgl-project/sglang) and [KTransformers](https://github.com/kvcache-ai/ktransformers).

It needs one 16 GB GPU, one CPU socket and 64 GB of RAM.

More RAM makes the decode faster: 1.7 to 2.1 times the 64 GB speed once every expert fits.

Where each part runs:

- **GPU:** attention and the dense layers, in their GGUF types, plus as many experts as its free
  memory holds.
- **CPU:** the other routed experts, computed from the GGUF's own rows by kt-kernel.
- **RAM:** a cache of as many experts as fit; the rest are read from the NVMe drive when the text
  routes to them.
- **Prefill:** the experts stream to the GPU and are multiplied there with llama.cpp's MMQ kernels.
- **Engram tables:** read from the Engram GGUF on disk.

This is the GGUF counterpart of
[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers),
which runs the MXFP4 checkpoint on 512 GB of RAM.
It carries the same SGLang and KTransformers changes, plus the GGUF ones in `patches/gguf/`.
It is pinned to SGLang main `65505981` and KTransformers `95009ea` (torch 2.14.1 with CUDA 13
wheels, FlashInfer 0.7.2, CUDA toolkit 13.1); later upstream versions are not tracked.

```
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --revision a419d74cd28d5b9f8c5e2b9077d11a609c17bc32 \
  --include "*IQ2_XXS*" "*Engram*" --local-dir /models/dsv41-gsq-iq2xxs
GGUF=/models/dsv41-gsq-iq2xxs ./setup.sh
GGUF=/models/dsv41-gsq-iq2xxs ./serve.sh
```

Or all of it in one command:

```
./run.sh
```

`run.sh` runs `setup.sh`, downloads the IQ2_XXS build and the Engram file into `./models/` when they are not there, and starts `serve.sh`.
`BUILD=IQ1_M` or `BUILD=IQ2_XS` picks another build, and `GGUF=<dir>` uses one already downloaded.
It remembers the directory in `state/run.env`, so the next `./run.sh` only starts the server.

The revision is the pack in llama.cpp's own layout (2026-10-27: six shards a build, the Engram tables as one tensor).
Downloads in the earlier layout (four shards a build) work too.
The shard-6 link the pack's README asks for is not needed here, and does no harm.

`GGUF=<dir> tools/check.sh` starts the server once, sends one short request, checks that an
answer comes back, and stops the server.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`, one request at a time.
`SGLANG_HOST=0.0.0.0` with `SGLANG_API_KEY=<key>` serves other machines (clients send
`Authorization: Bearer <key>`). The model is served as text only: the GGUF has no vision tower.

## What it needs

- **GPU: one, 16 GB, SM120 (RTX 50) as tested; an 8 GB RTX 20 (SM75) runs it too**
  ([below](#an-rtx-2070-8-gb-and-a-6-core-desktop-cpu)). The dense layers in their GGUF types take
  6.8 GB; the rest is the KV pool (131,072 tokens by default, 32,768 on a card under 12 GB;
  `SGLANG_CONTEXT_LENGTH`) and working room that the prefill chunk and the extra hot experts take. With several GPUs
  visible, `serve.sh` uses them all ([below](#two-gpus-and-two-sockets)).
  `CUDA_VISIBLE_DEVICES` or `KT_GGUF_TP` picks fewer.
- **CPU and RAM: the socket the GPU hangs off** (with several GPUs, each GPU's socket).
  `serve.sh` runs the experts on that NUMA node's physical cores and sizes the expert cache
  from its memory:
  - 64 GB of RAM is enough; more is faster. The cache holds the experts' GGUF rows (124 GiB for
    IQ2_XXS) as far as the node has room, less 18 GiB for everything else
    (`KT_GGUF_HOST_RESERVE_GIB`); what does not fit is read from the file again when it is
    routed, so decode then waits on the drive;
  - when the node holds every expert and the IQ1_M decode copy too (76 GiB more for IQ2_XXS),
    the copy is kept: an IQ1_M layer's CPU decode takes 0.55 ms instead of 0.94 ms on it;
  - a cgroup memory limit counts (`systemd-run --user --scope -p MemoryMax=64G ./serve.sh`
    runs it as a 64 GB machine).

  kt-kernel is built for the CPU it is built on; its GGUF kernels need AVX2.
- **Disk: NVMe, about 260 GB.** The IQ2_XXS build is 138 GB of shards and the 111 GB Engram
  file. The engram rows are read from the drive in decode, the experts RAM does not hold too,
  and the prefill streams the experts from it: with less RAM, the drive's speed sets more of
  the decode.
- **Software.** Linux, bash 4.4 or newer, the CUDA toolkit (built and run with 13.1), git,
  gcc/g++, the libnuma and hwloc headers (`libnuma-dev libhwloc-dev`), and `uv` or Python 3.12
  with its headers (`python3.12-dev`).
  The Python packages are pinned in `requirements.txt`.
- **Host settings**, as root: `vm.min_free_kbytes = 8388608` and swap off.

## What setup.sh does

1. Clones sgl-project/sglang at `65505981` and applies the MXFP4 recipe's patches (tiers 0 to
   2, [docs/patches.md](docs/patches.md), [docs/tier1.md](docs/tier1.md)), then
   `patches/gguf/sglang-gguf.patch` ([docs/gguf.md](docs/gguf.md)).
2. Clones kvcache-ai/ktransformers at `95009ea` and applies `patches/kt-kernel-tree.patch`,
   `patches/kt-kernel-llama-cpp.patch`, then `patches/gguf/kt-kernel-gguf.patch` (the GGUF expert
   cache, zero-copy experts, the IQ1_M decode copy) and `patches/gguf/kt-llamafile-iq2xs.patch`
   (IQ2_XS in kt-kernel's x86 kernels).
3. Creates `.venv` from `requirements.txt`, builds kt-kernel for this CPU and the GPU, and builds
   the CUDA kernels the server would otherwise compile on its first request.
4. For `GGUF=<dir>`, writes `state/gguf-<dir name>/`:
   - `config/`: the base model's `config.json`, tokenizer and chat encoding, downloaded from
     deepseek-ai/DeepSeek-V4.1-Flash-1017 (a few MB, no weights; `BASE=<dir>` takes them from a
     local copy), with `quantization_config` set to GGUF (`tools/make-gguf-config.py`);
   - `engram/engram-manifest.json`: where each engram table lies in the Engram GGUF;
   - `experts.json`: the routed experts' bytes, for the cache size;
   - `io-probe.json`: how fast the drive under the GGUF reads, and from how many threads
     (`tools/io-probe.py`, about a minute, reads only).

Each patch is checked against its pinned tree before it is applied. Re-running skips the
finished steps.

## What serve.sh does

At start it reads each GPU, its NUMA node, the node's cores and memory, and sets the CPU threads,
the expert cache and the CUDA arch from them. The first launch of a setup also calibrates the
prefill chunk (the server starts one extra time). Every default is in the script and can be
overridden by the variable named next to it. Logs go to `logs/<GGUF dir name>/<timestamp>/`.

- one expert a layer stays on the GPU, and the hot lend fills the KV pool's free pages and the
  driver's idle memory with more, swapped every 8 decode steps for the experts the text routes
  to most; the swapped experts' rows are read in the background and copied on a side stream
- the prefill lends the dense weights' memory and the KV pool's free pages to its chunk, and
  streams the experts' GGUF rows from the drive to the GPU
- the threads that read the drive run away from the ones computing the experts, sized from a
  measurement of the drive that `setup.sh` takes ([docs/gguf.md](docs/gguf.md#drive-reads))
- V4.1's tool-call and reasoning parsers, as in the MXFP4 recipe

## Results

AMD EPYC 7452 (Zen 2, AVX2, one of two sockets: 32 cores, 256 GB DDR4), one RTX 5070 Ti 16 GB.
The IQ2_XXS build sits on one NVMe drive.
The server is built from this repository and run with the `serve.sh` defaults on one GPU
(`CUDA_VISIBLE_DEVICES=0`; 131,072-token context, one request at a time).
`tools/bench.py` measures it.
Every column comes from one launch, after one warm-up pass of the same prompts.
The first pass fills the expert cache from the drive.

The 64 GB and 128 GB columns are that socket under a memory limit,
`systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 ./serve.sh` (or `128G`), not a
desktop of that size: a desktop's memory and drive will give other numbers. `serve.sh` reads the
limit and sizes the expert cache from it:

| RAM | expert cache | of the experts' 124 GiB | expert reads that hit it |
|---|---|---|---|
| 64 GB | 46 GiB | 37% | 90.2% |
| 128 GB | 110 GiB | 89% | 98.2% |
| 251 GiB (the socket) | 195 GiB, with the IQ1_M decode copy | all | 99.1% |

Decode, 1,024 tokens a reply, tok/s (`--other-text` for the rows below the first):

| text | 64 GB | 128 GB | 251 GiB |
|---|---|---|---|
| Japanese prose (median of four prompts) | 25.7 | 33.4 | 43.5 |
| JSON | 18.3 | 24.3 | 31.0 |
| Chinese prose | 17.9 | 28.6 | 34.1 |
| English prose | 17.3 | 25.2 | 31.1 |
| math | 17.3 | 24.8 | 31.2 |
| Python code | 13.6 | 21.5 | 27.6 |
| English technical prose | 13.3 | 21.2 | 27.9 |

Prefill, one prompt of random token ids:

| prompt (tokens) | 64 GB | 128 GB | 251 GiB |
|---|---|---|---|
| 35,000 | 1,268 tok/s (28 s) | 1,855 tok/s (19 s) | 2,010 tok/s (17 s) |

The misses go to the drive.
At 128 GB one expert read in fifty-five misses, and the decode is already 1.3 to 1.6 times the 64 GB one.
The rest of the way comes with the last 11% of the experts and the IQ1_M decode copy.

Outside prefill the hot lend holds 19 to 21 more experts a layer on the GPU.
With all experts in RAM that was 799 to 827 slots, in the 6.6 to 6.7 GiB the driver had free.
The slots are swapped for the experts the text routes to.
Japanese prose runs fastest: its routes gather on fewer experts, so the slots and the cache take
more of them.

### Two GPUs and two sockets

Both EPYC 7452 sockets (64 cores, 512 GB) and two RTX 5070 Ti 16 GB, the same drive and build.
`serve.sh` with no settings takes both GPUs, one rank on each socket.
The expert cache holds every expert and the IQ1_M decode copy (204 GiB).
How the two GPUs share the work is in [docs/gguf.md](docs/gguf.md#two-or-more-gpus).

| text | decode, tok/s |
|---|---|
| Japanese prose (median of four prompts) | 61.8 |
| English prose | 41.7 |
| JSON | 41.3 |
| math | 39.5 |
| Chinese prose | 38.2 |
| Python code | 36.6 |
| English technical prose | 36.6 |

A 35,000-token prompt prefills at 2,759 tok/s (13 s).

### An RTX 2070 8 GB and a 6-core desktop CPU

Intel Core i5-9600K (6 cores, AVX2, two-channel DDR4), one RTX 2070 8 GB (Turing, SM75), the pack on
a PCIe 3 NVMe drive (NTFS), the `serve.sh` defaults for that card (32,768-token context, 2,048-token
prefill chunk) under a 64 GB memory limit, on a desktop in use and without the host settings above.
What Turing takes is in [docs/gguf.md](docs/gguf.md#turing-sm75).

| | tok/s |
|---|---|
| decode, Japanese prose (the first bench prompt, 256 tokens, second pass) | 1.8 |
| decode, English prose (256 tokens, second pass) | 1.7 |
| prefill, 8,000 tokens | 34 |

The six CPU cores set the pace (`nvidia-smi` showed the GPU 8% busy in decode).

## Accuracy

The served model is scored against the base model's own reference implementation, DeepSeek's
`inference/model.py` run on the original FP8 checkpoint ([tools/ref-nll.py](tools/ref-nll.py)): both
score the same 24 pieces of 480 tokens from two public-domain texts in [tools/nll-text/](tools/nll-text/),
teacher-forced.

| run | mean NLL | minus the reference (SE over the 24 pieces) |
|---|---|---|
| reference (`inference/model.py`, the FP8 checkpoint) | 0.7185 | |
| served, the IQ2_XXS GGUF | 1.0479 | +0.329 (0.054) |
| served, the IQ2_XXS GGUF on the RTX 2070 (`KT_GPU_EXPERTS=0`) | 1.0468 | +0.328 (0.053) |
| served, the IQ2_XXS GGUF on two GPUs and two sockets | 1.0485 | +0.330 (0.054) |

The difference is the GGUF's quantization, not the serving: served with only the routed experts
from the GGUF (the dense layers and the engram from the FP8 checkpoint) the NLL is +0.312 above the
reference; the dense layers in their GGUF types add 0.007 and the IQ4_NL engram tables 0.006. The
GGUF's experts match the checkpoint's expert by expert (cosine 0.93 for IQ2_XXS tensors, 0.86 for
IQ1_M) with a relative weight error of 36% and 51%. The same serving path on the MXFP4 checkpoint
lands within 0.001 of the reference. The texts are well known, so the NLL is low; they compare two
runs on the same tokens, not the model's quality, which the pack's model card reports on benchmarks.

On an 8 GB card the scoring needs `KT_GPU_EXPERTS=0` at launch: the logprobs of a whole piece take
the room the resident expert holds. The dynamic hot slots move the NLL by a few thousandths between
launches (which experts the GPU computes changes the arithmetic); with the slots fixed
(`KT_KV_HOT_DYNAMIC_SLOTS=0`) two launches give the same bits.

To repeat it (the reference needs the FP8 checkpoint and one 16 GB GPU, about 4 minutes):

```
python tools/ref-nll.py <FP8 checkpoint dir> ref.json
GGUF=<GGUF dir> ./serve.sh --background
python tools/nll.py score state/gguf-<GGUF dir name>/config served.json
python tools/nll.py compare ref.json served.json
```

## Not done

- Other GPUs than SM120 and SM75: the GGUF kernels build for the GPU's own arch, but only an RTX
  5070 Ti and an RTX 2070 were run. V100 (SM70) would take a CUDA 12 build of the stack (CUDA 13
  has no SM70).
- Two GPUs of different architectures: should work, not run.
- The IQ1_M and IQ2_XS builds of the pack: their types are all supported, but only IQ2_XXS was
  measured.
- CPUs other than Zen 2 and Coffee Lake (i5-9600K).

## License

Apache-2.0 (see `LICENSE`), except the files `NOTICE` lists: the engram adapter (MIT, from
0xSero's adapter), llama.cpp's MMQ and quantization code vendored by `patches/gguf/sglang-gguf.patch`
(MIT, ggml authors), and the IQ2_XS kernel in `patches/gguf/kt-llamafile-iq2xs.patch` (MIT, from
ik_llama.cpp). The GGUF weights are ISTA-DASLab's (MIT) and the model DeepSeek's (see the model card).
