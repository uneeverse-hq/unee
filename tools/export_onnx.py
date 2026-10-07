"""Put fine-tuned weights into onnx-community's Qwen3.5-0.8B text ONNX graph, for transformers.js (WebGPU/WASM).

    PYTHONUTF8=1 .venv/Scripts/python tools/export_onnx.py --merged models/unee-v1/merged --out models/onnx-unee-v1
    PYTHONUTF8=1 .venv/Scripts/python tools/export_onnx.py --check     # re-quantize the base weights, compare bytes

The graph (onnx-community/Qwen3.5-0.8B-Text-ONNX, model_q4f16.onnx) stores every linear layer as a 4-bit
MatMulNBits weight (asymmetric, block 32). LoRA only changes linear layers, so each one is re-quantized from the
merged model with onnxruntime's own block quantizer, under the same names; norms, convolutions and the tied
embedding are left as they are. `--check` re-quantizes the *base* model and compares with the published bytes.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "onnx-base"
NAME = re.compile(r"^model_layers_(\d+)_(gdn|attn|mlp)_(\w+?)_MatMul_weight_quant$")
MODULE = {"gdn": "linear_attn", "attn": "self_attn", "mlp": "mlp"}


def hf_key(onnx_weight: str, prefix: str) -> str | None:
    m = NAME.match(onnx_weight)
    if not m:
        return None
    layer, module, proj = m.groups()
    return f"{prefix}layers.{layer}.{MODULE[module]}.{proj}.weight"


def quantize(w_out_in: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ORT's MatMulNBits 4-bit asymmetric block quantization of a [out, in] torch weight (B = W^T is [K, N])."""
    b = np.ascontiguousarray(w_out_in.T.astype(np.float32))
    k, n = b.shape
    k_blocks = (k + block - 1) // block
    packed = np.zeros((n, k_blocks, block // 2), dtype=np.uint8)
    zp = np.zeros(n * ((k_blocks + 1) // 2), dtype=np.uint8)
    scales = np.zeros(n * k_blocks, dtype=np.float32)
    quantize_matmul_4bits(packed, b, scales, zp, block, n, k, False)
    return packed, scales, zp


def load_weights(path: str) -> tuple[dict, str]:
    files = sorted(glob.glob(str(Path(path) / "*.safetensors")))
    out = {}
    for f in files:
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if "proj" in k and k.endswith(".weight") and "visual" not in k and "mtp" not in k:
                    out[k] = sf.get_tensor(k).float().numpy()
    prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in out) else "model."
    return out, prefix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", help="merged Hugging Face model directory")
    ap.add_argument("--out", help="output directory (config, tokenizer and onnx/ are written there)")
    ap.add_argument("--check", action="store_true", help="re-quantize the base model and compare with the graph")
    ap.add_argument("--variant", nargs="+", default=["q4f16"], choices=("q4f16", "q4"),
                    help="q4f16 for WebGPU, q4 (fp32 scales) for the WASM/CPU fallback")
    args = ap.parse_args()

    for variant in args.variant:
        export(args, variant)


def export(args, variant: str) -> None:
    src = BASE / "onnx" / f"model_{variant}.onnx"
    model = onnx.load(str(src))  # loads the external weights too
    inits = {t.name: t for t in model.graph.initializer}
    if args.check:
        snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
        weights, prefix = load_weights(snap)
    else:
        weights, prefix = load_weights(args.merged)

    done = 0
    same_bytes = total_bytes = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        qname, sname, zname = node.input[1], node.input[2], node.input[3]
        key = hf_key(qname, prefix)
        if key is None:  # the lm_head, tied to the embedding, which training does not change
            continue
        block = next(a.i for a in node.attribute if a.name == "block_size")
        packed, scales, zp = quantize(weights[key], block)
        old_q = numpy_helper.to_array(inits[qname])
        old_s = numpy_helper.to_array(inits[sname])
        old_z = numpy_helper.to_array(inits[zname])
        packed = packed.reshape(old_q.shape)
        zp = zp.reshape(old_z.shape)
        scales = scales.reshape(old_s.shape).astype(old_s.dtype)
        if args.check:
            same_bytes += int((packed == old_q).sum()) + int((zp == old_z).sum())
            total_bytes += packed.size + zp.size
        else:
            for name, arr in ((qname, packed), (sname, scales), (zname, zp)):
                inits[name].CopyFrom(numpy_helper.from_array(arr, name))
        done += 1
    if args.check:
        print(f"{done} linear layers re-quantized from the base model; {same_bytes / total_bytes:.4%} of the packed "
              f"bytes match the published graph")
        return
    out = Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    # onnx.save appends external tensors to an existing data file, so re-exporting into the same folder silently
    # doubled the download (469 MB -> 939 MB). Start from an empty file every time.
    (out / "onnx" / f"model_{variant}.onnx_data").unlink(missing_ok=True)
    onnx.save(model, str(out / "onnx" / f"model_{variant}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"model_{variant}.onnx_data", size_threshold=1024)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(BASE / f, out / f)
    gen = json.loads((out / "generation_config.json").read_text())
    gen["eos_token_id"] = [248046, 248044]  # <|im_end|> ends an assistant turn; the base config only lists <|endoftext|>
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
    print(f"{done} linear layers replaced; wrote {out}")


if __name__ == "__main__":
    main()
