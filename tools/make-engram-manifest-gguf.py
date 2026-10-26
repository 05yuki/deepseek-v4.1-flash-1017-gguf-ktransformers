#!/usr/bin/env python3
"""An engram manifest for the V4.1 recipe's adapter from a llama.cpp "deepseek41" GGUF: each blk.<L>.engram_embd
table (IQ4_NL, [256, rows] in ggml order) as layer L, read in place from the GGUF file (row_bytes 144).
usage: make-engram-manifest-gguf.py <gguf dir> <out dir>"""
import glob, json, os, re, sys

from gguf import GGUFReader

gguf_dir, out_dir = sys.argv[1], sys.argv[2]
layers = {}
for path in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
    r = GGUFReader(path)
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.engram_embd\.weight$", t.name)
        if not m:
            continue
        dim, rows = (int(x) for x in t.shape.tolist())  # ggml order: [256, rows]
        assert t.tensor_type.name == "IQ4_NL" and dim == 256, (t.name, t.tensor_type.name, dim)
        assert t.n_bytes == rows * 144, (t.name, t.n_bytes, rows)
        layers[m.group(1)] = {
            "file": os.path.basename(path), "rows": rows, "dim": dim, "blocks": dim // 32, "block_size": 32,
            "weight_dtype": "IQ4_NL", "row_bytes": 144, "scale_dtype": None,
            "weight_offset": int(t.data_offset), "scale_offset": 0, "weight_bytes": int(t.n_bytes), "scale_bytes": 0,
        }
assert layers, f"no blk.N.engram_embd tables in {gguf_dir}"
os.makedirs(out_dir, exist_ok=True)
manifest = {"source": os.path.abspath(gguf_dir), "base_dir": os.path.abspath(gguf_dir), "layers": layers}
with open(os.path.join(out_dir, "engram-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"{out_dir}/engram-manifest.json: layers {sorted(layers, key=int)} from {gguf_dir}")
