# DeepSeek-V4.1-Flash-1017 GGUF on SGLang + KTransformers (one GPU, one CPU socket)

Run [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on one 16 GB GPU and the CPU socket next to it. The 384 routed experts per layer run on the
CPU from the GGUF's own rows through [KTransformers](https://github.com/kvcache-ai/ktransformers)'
kt-kernel (its LLAMAFILE backend), held in RAM as far as the node has room. Attention and the
dense layers run on the GPU in their GGUF types through [SGLang](https://github.com/sgl-project/sglang),
and the GPU's free memory holds as many more experts as fit, swapped for what the text routes to.
The prefill streams the experts to the GPU and multiplies them there with llama.cpp's MMQ kernels.
The engram tables are read from the Engram GGUF on disk.

This repository is the GGUF counterpart of
[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)
(the MXFP4 checkpoint on 512 GB of RAM): the same SGLang and KTransformers changes, plus the GGUF
ones in `patches/gguf/`. It is a snapshot against SGLang main `65633129` and KTransformers
`95009ea` (torch 2.14.1 with CUDA 13 wheels, FlashInfer 0.7.2, CUDA toolkit 13.1); later
upstream versions are not tracked.

```
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ2_XXS*" "*Engram*" --local-dir /models/dsv41-gsq-iq2xxs
GGUF=/models/dsv41-gsq-iq2xxs ./setup.sh
GGUF=/models/dsv41-gsq-iq2xxs ./serve.sh
```

`GGUF=<dir> tools/check.sh` starts the server once, sends one short request, checks that an
answer comes back, and stops the server.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`, one request at a time.
`SGLANG_HOST=0.0.0.0` with `SGLANG_API_KEY=<key>` serves other machines (clients send
`Authorization: Bearer <key>`). The model is served as text only: the GGUF has no vision tower.

## What it needs

- **GPU: one, 16 GB, SM120 (RTX 50) as tested.** The dense layers in their GGUF types take
  8.3 GB; the rest is the KV pool (131,072 tokens by default, `SGLANG_CONTEXT_LENGTH`) and
  working room that the prefill chunk and the extra hot experts take. With several GPUs
  visible, `serve.sh` takes the first (`CUDA_VISIBLE_DEVICES` picks another): the GGUF
  experts do not split across tensor-parallel ranks.
- **CPU and RAM: the socket the GPU hangs off.** `serve.sh` runs the experts on that NUMA
  node's physical cores and sizes the expert cache from its memory:
  - the cache holds the experts' GGUF rows (124 GiB for IQ2_XXS) as far as the node has room,
    less 18 GiB for everything else (`KT_GGUF_HOST_RESERVE_GIB`); what does not fit is read
    from the file again when it is routed, so decode then waits on the drive;
  - when the node holds every expert and the IQ1_M decode copy too (76 GiB more for IQ2_XXS),
    the copy is kept: an IQ1_M layer's CPU decode takes 0.55 ms instead of 0.94 ms on it;
  - a cgroup memory limit counts (`systemd-run --user --scope -p MemoryMax=64G ./serve.sh`
    runs it as a 64 GB machine).

  kt-kernel is built for the CPU it is built on; its GGUF kernels need AVX2.
- **Disk: NVMe, about 260 GB.** The IQ2_XXS build is 138 GB of shards and the 111 GB Engram
  file. The engram rows are read from the drive in decode, and the prefill streams the experts
  from it.
- **Software.** Linux, bash 4.4 or newer, the CUDA toolkit (built and run with 13.1), git,
  gcc/g++, the libnuma and hwloc headers (`libnuma-dev libhwloc-dev`), and `uv` or Python 3.12.
  The Python packages are pinned in `requirements.txt`.
- **Host settings**, as root: `vm.min_free_kbytes = 8388608` and swap off.

## What setup.sh does

1. Clones sgl-project/sglang at `65633129` and applies the MXFP4 recipe's patches (tiers 0 to
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
   - `experts.json`: the routed experts' bytes, for the cache size.

Each patch is checked against its pinned tree before it is applied. Re-running skips the
finished steps.

## What serve.sh does

At start it reads the GPU, its NUMA node, the node's cores and memory, and sets the CPU threads,
the expert cache and the CUDA arch from them. The first launch of a setup also calibrates the
prefill chunk (the server starts one extra time). Every default is in the script and can be
overridden by the variable named next to it. Logs go to `logs/<GGUF dir name>/<timestamp>/`.

- one expert a layer stays on the GPU, and the hot lend fills the KV pool's free pages and the
  driver's idle memory with more, swapped every 8 decode steps for the experts the text routes
  to most; the swapped experts' rows are read in the background and copied on a side stream
- the prefill lends the dense weights' memory and the KV pool's free pages to its chunk, and
  streams the experts' GGUF rows from the drive to the GPU
- V4.1's tool-call and reasoning parsers, as in the MXFP4 recipe

## Results

AMD EPYC 7452 (Zen 2, AVX2, one of two sockets: 32 cores, 256 GB DDR4), one RTX 5070 Ti 16 GB, the
IQ2_XXS build on NVMe; built from this repository and run with the `serve.sh` defaults (131,072-token
context, one request at a time), measured with `tools/bench.py`, every table from one launch, after
one warm-up pass of the same prompts (the first pass fills the expert cache from the drive).

Decode, 1,024 tokens a reply (`--other-text` for the rows below the first):

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 39.9 |
| Chinese prose | 31.5 |
| math | 30.3 |
| English prose | 29.2 |
| JSON | 28.9 |
| English technical prose | 27.1 |
| Python code | 26.7 |

Prefill, one prompt of random token ids:

| prompt (tokens) | tok/s | time |
|---|---|---|
| 35,000 | 1,650 | 21 s |

The node's 251 GiB take the whole IQ2_XXS build: the expert cache holds every expert and the IQ1_M
decode copy (195 GiB of its 204), and the server process uses 220 GB of the node. Outside prefill
the hot lend holds 15-16 more experts a layer on the GPU (610-640 slots, in 5 GiB the driver had
free), swapped for what the text routes to; Japanese prose runs fastest because its routes gather on
fewer experts, so the slots take more of them.

### 64 GB of RAM

The same launch under `systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0`: `serve.sh`
reads the limit and keeps a 46 GiB expert cache of raw rows (no IQ1_M copy), about 37% of the experts;
the rest is read from the drive when it is routed (90% of the expert reads hit the cache).

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 22.7 |
| math | 18.1 |
| Chinese prose | 17.4 |
| JSON | 17.1 |
| English prose | 16.2 |
| English technical prose | 13.0 |
| Python code | 12.8 |

| prompt (tokens) | tok/s | time |
|---|---|---|
| 35,000 | 913 | 38 s |

## Accuracy

The served model is scored against the base model's own reference implementation, DeepSeek's
`inference/model.py` run on the original FP8 checkpoint ([tools/ref-nll.py](tools/ref-nll.py)): both
score the same 24 pieces of 480 tokens from two public-domain texts in [tools/nll-text/](tools/nll-text/),
teacher-forced.

| run | mean NLL | minus the reference (SE over the 24 pieces) |
|---|---|---|
| reference (`inference/model.py`, the FP8 checkpoint) | 0.7185 | |
| served, the IQ2_XXS GGUF | 1.0479 | +0.329 (0.054) |

The difference is the GGUF's quantization, not the serving: served with only the routed experts
from the GGUF (the dense layers and the engram from the FP8 checkpoint) the NLL is +0.312 above the
reference; the dense layers in their GGUF types add 0.007 and the IQ4_NL engram tables 0.006. The
GGUF's experts match the checkpoint's expert by expert (cosine 0.93 for IQ2_XXS tensors, 0.86 for
IQ1_M) with a relative weight error of 36% and 51%. The same serving path on the MXFP4 checkpoint
lands within 0.001 of the reference. The texts are well known, so the NLL is low; they compare two
runs on the same tokens, not the model's quality, which the pack's model card reports on benchmarks.

To repeat it (the reference needs the FP8 checkpoint and one 16 GB GPU, about 4 minutes):

```
python tools/ref-nll.py <FP8 checkpoint dir> ref.json
GGUF=<GGUF dir> ./serve.sh --background
python tools/nll.py score state/gguf-<GGUF dir name>/config served.json
python tools/nll.py compare ref.json served.json
```

## Not done

- Other GPUs than SM120: the GGUF kernels build for the GPU's own arch, but only an RTX 5070 Ti
  was run.
- More than one GPU: not supported (the GGUF expert blocks do not split across ranks).
- The IQ1_M and IQ2_XS builds of the pack: their types are all supported, but only IQ2_XXS was
  measured.
- CPUs other than Zen 2.

## License

Apache-2.0 (see `LICENSE`), except the files `NOTICE` lists: the engram adapter (MIT, from
0xSero's adapter), llama.cpp's MMQ and quantization code vendored by `patches/gguf/sglang-gguf.patch`
(MIT, ggml authors), and the IQ2_XS kernel in `patches/gguf/kt-llamafile-iq2xs.patch` (MIT, from
ik_llama.cpp). The GGUF weights are ISTA-DASLab's (MIT) and the model DeepSeek's (see the model card).
