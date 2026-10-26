#!/usr/bin/env bash
# Build everything serve.sh needs for a GGUF build of DeepSeek-V4.1-Flash-1017, inside this directory:
#   third_party/sglang         sgl-project/sglang main @ 66f79f1a + patches/sglang-tier0..2 + patches/gguf/sglang-gguf.patch
#   third_party/ktransformers  kvcache-ai/ktransformers @ 95009ea + patches/kt-kernel-*.patch + patches/gguf/kt-*.patch
#   .venv                      Python 3.12, requirements.txt, kt-kernel built for this CPU and GPU
#   engram-adapter/librow_store.so, build/libkv_vmm.so, the GGUF CUDA kernels (.cache/torch_extensions)
#   state/gguf-<name>/         for GGUF=<dir>: config/ (config.json and tokenizer for SGLang),
#                              engram/engram-manifest.json, experts.json (the routed experts' bytes)
#
# usage: GGUF=<dir with the GGUF shards and the Engram file> [BASE=<dir>] ./setup.sh
#
# GGUF is a directory holding one build of ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (its shards
# and DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf side by side, as the pack's README downloads them). SGLang
# reads the architecture and the tokenizer from deepseek-ai/DeepSeek-V4.1-Flash-1017: BASE is a directory
# with its config.json, tokenizer and encoding/; without BASE those few files are downloaded (no weights).
# Needs: git, a C/C++ compiler, the CUDA toolkit (CUDA_HOME, default /usr/local/cuda;
# built and run here with 13.1), libnuma and hwloc headers (Debian/Ubuntu: libnuma-dev
# libhwloc-dev), and either uv or python3.12 with its headers (python3.12-dev). Re-running skips the
# finished steps.
# Knobs: CUDA_HOME; KT_CUDA_ARCHS (default: the visible GPUs' compute capability);
# CPUINFER_CPU_INSTRUCT (kt-kernel's CPU target, default NATIVE: this machine's ISA);
# CMAKE_ARGS (extra CMake flags for kt-kernel, space separated); headers and libraries
# outside the system paths go in through CPLUS_INCLUDE_PATH / LIBRARY_PATH.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
SGLANG_COMMIT=66f79f1a9070ea7ae8f8e5d9b3381340ea0ec83e
KT_COMMIT=95009ea6856c0799e517e93cb12be5e8494bc7ce
# the GGUF patch sits on the whole of the MXFP4 recipe's SGLang changes (tiers 0 to 2, every tier-1 feature)
TIER=2
sglang_patches=()
for f in "$here"/patches/sglang-tier0/[0-9][0-9]-*.patch; do sglang_patches+=("sglang-tier0/$(basename "$f")"); done
features=()
for f in "$here"/patches/sglang-tier1/[0-9][0-9]-*.patch; do
  slug="$(basename "$f" .patch)"; features+=("${slug#[0-9][0-9]-}")
  sglang_patches+=("sglang-tier1/$(basename "$f")")
done
sglang_patches+=(sglang-tier2-no-p2p.patch gguf/sglang-gguf.patch)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p third_party build state
step() { printf '\n== %s\n' "$*"; }
# a step's marker file holds what it built: the tier or variant and a hash of the patches, so a
# changed patch is applied (and kt-kernel rebuilt) again
built_as() { [[ -f "$1" && "$(cat "$1")" == "$2" ]]; }
patch_hash() { (cd "$here/patches" && cat "$@") | sha256sum | cut -c1-12; }
sglang_state="$TIER $(patch_hash "${sglang_patches[@]}")"
kt_variant="patched gguf $(patch_hash kt-kernel-tree.patch kt-kernel-llama-cpp.patch kt-kernel-load-prefetch.py gguf/kt-kernel-gguf.patch gguf/kt-llamafile-iq2xs.patch)"
# a patch is checked against the pinned tree first; nothing is applied when it does not fit
apply_checked() {
  local tree="$1" patch="$2"
  if ! git -C "$tree" apply --check --whitespace=nowarn "$patch"; then
    echo "$patch does not apply to $tree; the tree was left as checked out" >&2
    exit 4
  fi
  git -C "$tree" apply --whitespace=nowarn "$patch"
}

step "SGLang @ ${SGLANG_COMMIT:0:8}, tiers 0-2 and GGUF"
if [[ ! -d third_party/sglang/.git ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git third_party/sglang
fi
if ! built_as third_party/sglang/.dsv41-1017-patched "$sglang_state"; then
  # a commit on sgl-project/sglang's main branch
  git -C third_party/sglang fetch --quiet origin "$SGLANG_COMMIT"
  git -C third_party/sglang checkout --quiet --force "$SGLANG_COMMIT"
  git -C third_party/sglang clean -fdq
  for p in "${sglang_patches[@]}"; do
    apply_checked third_party/sglang "$here/patches/$p"
  done
  echo "$sglang_state" > third_party/sglang/.dsv41-1017-patched
  printf '%s\n' "${features[@]}" > third_party/sglang/.dsv41-1017-features
fi

step "KTransformers @ ${KT_COMMIT:0:8}, $kt_variant"
if [[ ! -d third_party/ktransformers/.git ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git third_party/ktransformers
fi
if ! built_as third_party/ktransformers/.dsv41-1017-patched "$kt_variant"; then
  git -C third_party/ktransformers checkout --quiet --force "$KT_COMMIT"
  git -C third_party/ktransformers clean -fdq
  git -C third_party/ktransformers submodule update --init --force --quiet third_party/llama.cpp third_party/pybind11
  git -C third_party/ktransformers/third_party/llama.cpp clean -fdq
  apply_checked third_party/ktransformers "$here/patches/kt-kernel-tree.patch"
  apply_checked third_party/ktransformers/third_party/llama.cpp "$here/patches/kt-kernel-llama-cpp.patch"
  # the GGUF experts: kt-kernel's LLAMAFILE backend, and IQ2_XS in its x86 kernels
  apply_checked third_party/ktransformers "$here/patches/gguf/kt-kernel-gguf.patch"
  apply_checked third_party/ktransformers "$here/patches/gguf/kt-llamafile-iq2xs.patch"
  echo "$kt_variant" > third_party/ktransformers/.dsv41-1017-patched
fi

step "Python environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 --seed .venv
  else
    python3.12 -m venv .venv
  fi
fi
py="$here/.venv/bin/python"
pip_install() {
  if command -v uv >/dev/null; then
    uv pip install --python "$py" "$@"
  else
    "$py" -m pip install "$@"
  fi
}
if [[ ! -f .venv/.requirements-done ]] || [[ requirements.txt -nt .venv/.requirements-done ]]; then
  # torch and its CUDA 13 wheels come from PyTorch's index, the rest from PyPI
  if command -v uv >/dev/null; then
    pip_install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  else
    pip_install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  fi
  touch .venv/.requirements-done
fi
# the patched SGLang tree (serve.sh also puts it first on PYTHONPATH)
if ! "$py" -c "import sglang, sys; sys.exit(0 if sglang.__file__.startswith('$here/third_party/sglang') else 1)" 2>/dev/null; then
  # the Rust extensions (router, gRPC) are not used here and would need cargo
  SGLANG_BUILD_RUST_EXTS=none pip_install --no-deps -e third_party/sglang/python
fi

step "kt-kernel (CPU experts)"
site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if ! built_as build/.kt-kernel-done "$kt_variant"; then
  if [[ -z "${KT_CUDA_ARCHS:-}" ]] && command -v nvidia-smi >/dev/null; then
    KT_CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
  fi
  export CPUINFER_CUDA_ARCHS="${KT_CUDA_ARCHS:-80;86;89;90;120}"
  export CPUINFER_CPU_INSTRUCT="${CPUINFER_CPU_INSTRUCT:-NATIVE}"
  # the CUDA side (stream memops, the streamed prefill's helpers)
  export CPUINFER_USE_CUDA="${CPUINFER_USE_CUDA:-1}"
  export PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH"
  echo "CUDA archs $CPUINFER_CUDA_ARCHS, CPU target $CPUINFER_CPU_INSTRUCT"
  pip_install --no-deps --no-build-isolation third_party/ktransformers/kt-kernel
  # each expert layer is read into the page cache in parallel before kt-kernel copies it
  "$py" patches/kt-kernel-load-prefetch.py "$site"
  echo "$kt_variant" > build/.kt-kernel-done
fi

# The streamed prefill's CUTLASS MoE is a FlashInfer JIT module. Built inside the server it would
# compile on the first long prompt, while the expert arenas fill the NUMA nodes, and a node-local OOM
# kills the scheduler before the build finishes; so it is built here, once per FlashInfer version and
# GPU arch, into the cache serve.sh uses.
if command -v nvidia-smi >/dev/null; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  fi_backend="${cap/./}"
  case "$cap" in 9.0|10.0|10.3|12.0|12.1) fi_arch="${cap}a" ;; *) fi_arch="$cap" ;; esac
  step "FlashInfer CUTLASS MoE for sm $cap (streamed prefill)"
  FLASHINFER_WORKSPACE_BASE="${SGLANG_CACHE_DIR:-$here/.cache/sglang}" \
  FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$fi_arch}" MAX_JOBS="${MAX_JOBS:-16}" \
  PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH" \
    "$py" -c "
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module as g
try:
    g('$fi_backend')
except ValueError as e:
    # FlashInfer has no CUTLASS MoE for this arch; the GGUF prefill runs on MMQ and does not need it
    print(f'not built: {e}')
else:
    print('built (or cached)')"
fi

step "engram row store (engram-adapter/librow_store.so)"
if [[ ! -f engram-adapter/librow_store.so || engram-adapter/row_store.cpp -nt engram-adapter/librow_store.so ]]; then
  g++ -O2 -std=c++17 -shared -fPIC -pthread engram-adapter/row_store.cpp -o engram-adapter/librow_store.so
fi

step "kv-vmm (build/libkv_vmm.so)"
if [[ ! -f build/libkv_vmm.so || kv-vmm/kv_vmm.c -nt build/libkv_vmm.so ]]; then
  gcc -O2 -Wall -shared -fPIC kv-vmm/kv_vmm.c -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda -o build/libkv_vmm.so
fi
# the KV pool module looks for the library beside itself first
cp build/libkv_vmm.so third_party/sglang/python/sglang/srt/layers/moe/libkv_vmm.so

# The GGUF kernels (the experts' MMQ for the streamed prefill and the hot slots, the decode's vector
# kernels) are torch extensions compiled on first use; built here for the same reason as above.
if command -v nvidia-smi >/dev/null; then
  step "GGUF CUDA kernels"
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$here/.cache/torch_extensions}" MAX_JOBS="${MAX_JOBS:-16}" \
  PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH" "$py" -c "
from sglang.srt.layers.moe import kt_gguf_moe_mmq, kt_gguf_moe_multi
kt_gguf_moe_mmq._load()
kt_gguf_moe_multi._load()
kt_gguf_moe_multi._load(per_route=True)
# below SM80: sgl-kernel's ops for this GPU's arch (the wheel has sm_80+ only)
from sglang.srt.layers import kt_sm75_ops
kt_sm75_ops.load_if_needed()
print('built (or cached)')"
fi

if [[ -n "${GGUF:-}" ]]; then
  gguf_dir="$(cd "$GGUF" && pwd)"
  out="state/gguf-$(basename "$gguf_dir")"
  step "SGLang config for $gguf_dir"
  if [[ ! -f "$out/config/config.json" ]]; then
    base="${BASE:-}"
    if [[ -z "$base" ]]; then
      # the base model's config, tokenizer and chat encoding only
      base="state/base-DeepSeek-V4.1-Flash-1017"
      "$here/.venv/bin/hf" download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$base" \
        --include config.json --include tokenizer.json --include tokenizer_config.json \
        --include "encoding/*" --include "inference/*"
    fi
    "$py" tools/make-gguf-config.py "$base" "$out/config"
  fi
  step "engram manifest and expert sizes for $gguf_dir"
  [[ -f "$out/engram/engram-manifest.json" ]] || "$py" tools/make-engram-manifest-gguf.py "$gguf_dir" "$out/engram"
  [[ -f "$out/experts.json" ]] || "$py" tools/gguf-expert-bytes.py "$gguf_dir" "$out/experts.json"
fi

step "check: the attention-output FP8 kernel on this GPU"
"$py" tools/dsv4-o-fp8-qdq-test.py 2>&1 | grep -v -i warn | tail -3

printf '\nDone. Start the server with: GGUF=<the same dir> ./serve.sh\n'
