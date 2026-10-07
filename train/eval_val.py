"""Accuracy and calibration on our own held-out templates (data/generated/val.jsonl), in the compact format.

    PYTHONUTF8=1 .venv/Scripts/python train/eval_val.py --model models/unee-v0/merged
    PYTHONUTF8=1 .venv/Scripts/python train/eval_val.py --model Qwen/Qwen3.5-0.8B --temperature 1.0

Used to choose training settings and the calibration temperature, so DecideBench is only run on finished models.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from unee.prompt import LETTERS, build_messages  # noqa: E402


def load_model(path: str, load_4bit: bool, dtype, device: str):
    """bf16 on the device, or 4-bit (bitsandbytes) when the full model does not fit next to other GPU users."""
    if load_4bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, quantization_config=q,
                                                    device_map={"": 0}).eval()
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()


def ece(points: list[tuple[float, bool]], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        bucket = [(p, c) for p, c in points if b / bins < p <= (b + 1) / bins or (b == 0 and p == 0)]
        if bucket:
            total += len(bucket) / len(points) * abs(sum(p for p, _ in bucket) / len(bucket)
                                                     - sum(c for _, c in bucket) / len(bucket))
    return total


@torch.inference_mode()
def letter_logits(model, tok, rec: dict, zero_shot: bool) -> torch.Tensor:
    shots = [] if zero_shot else [(e["state"], e["gold"]) for e in rec["examples"]]
    msgs = build_messages(rec["state"], rec["question"], rec["options"], shots)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    lids = [tok.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(rec["options"]))]
    return logits[lids].cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default=str(ROOT / "data" / "generated" / "val.jsonl"))
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--load-4bit", action="store_true", help="load the model in 4-bit (big models on small GPUs)")
    ap.add_argument("--out", help="also write the result JSON here")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.val).read_text(encoding="utf-8").split("\n") if l.strip()]
    recs = recs[: args.limit] if args.limit else recs
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.load_4bit, torch.bfloat16, "cuda")
    rows = []
    for r in recs:
        keys = [o["key"] for o in r["options"]]
        rows.append((letter_logits(model, tok, r, args.zero_shot), keys.index(r["gold"]), r))

    def score(temp: float) -> dict:
        pts, nll, fam = [], 0.0, defaultdict(list)
        for logits, gold, r in rows:
            p = torch.softmax(logits / temp, -1)
            pred = int(p.argmax())
            pts.append((float(p.max()), pred == gold))
            nll -= math.log(max(float(p[gold]), 1e-9))
            fam[r["family"]].append(pred == gold)
        return {"acc": sum(c for _, c in pts) / len(pts), "ece": ece(pts), "nll": nll / len(pts),
                "worst_families": sorted(((round(sum(v) / len(v), 3), k) for k, v in fam.items()))[:6]}

    base = score(1.0)
    best_t = min((t / 20 for t in range(10, 61)), key=lambda t: score(t)["nll"])  # temperature 0.5..3.0
    out = {"model": args.model, "n": len(rows), "zero_shot": args.zero_shot, "t1": base,
           "best_temperature": best_t, "at_best_t": score(best_t)}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
