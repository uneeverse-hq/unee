"""Run a Hugging Face causal LM on DecideBench on this machine, scoring options from next-token logits.

    .venv/Scripts/python bench/run_local.py --model Qwen/Qwen3.5-0.8B --device cuda
    .venv/Scripts/python bench/run_local.py --model Qwen/Qwen3.5-0.8B --device cpu --threads 4 --limit 40

Uses DecideBench's own chat prompt (TEV's system prompt, one worked example per option as earlier turns) and reads
the probability of each option's letter at the first answer position: one forward pass per item, no sampling.
Results go to bench/results/<name>.jsonl in DecideBench's Prediction format, so its scorer reads them unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "decidebench"))
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from decidebench import fewshot  # noqa: E402
from decidebench.dataset import Item, load_items, validate  # noqa: E402
from decidebench.prompts import LETTERS, build_messages  # noqa: E402
from unee.prompt import build_messages as compact_messages  # noqa: E402
from decidebench.score import load_rows, summarize  # noqa: E402
from decidebench.types import FREE, Prediction  # noqa: E402

RESULTS = HERE / "results"


def load_model(path: str, load_4bit: bool, dtype, device: str):
    """bf16 on the device, or 4-bit (bitsandbytes) when the full model does not fit next to other GPU users."""
    if load_4bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, quantization_config=q,
                                                    device_map={"": 0}).eval()
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()


def item_messages(item: Item, zero_shot: bool, fmt: str) -> list[dict]:
    """DecideBench's chat prompt, or Unee's compact one; both get the item's seeded worked examples."""
    shots = [] if zero_shot else fewshot.for_item(item)
    if fmt == "compact":
        return compact_messages(item.state, item.question, [{"key": o.key, "description": o.description}
                                                             for o in item.options], [(e.state, e.gold) for e in shots])
    return build_messages(item, shots)


def letter_ids(tok) -> list[int]:
    ids = []
    for letter in LETTERS[:8]:
        enc = tok.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter!r} is {len(enc)} tokens for this tokenizer")
        ids.append(enc[0])
    return ids


def prompt_ids(tok, item: Item, zero_shot: bool, fmt: str) -> torch.Tensor:
    msgs = item_messages(item, zero_shot, fmt)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


@torch.inference_mode()
def predict(model, tok, lids: list[int], item: Item, name: str, device: str, zero_shot: bool, fmt: str,
            temperature: float = 1.0) -> Prediction:
    ids = prompt_ids(tok, item, zero_shot, fmt).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    if device == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000
    n = len(item.options)
    p = torch.softmax(logits[lids[:n]] / temperature, dim=-1).tolist()
    probs = {o.key: p[i] for i, o in enumerate(item.options)}
    key = max(probs, key=probs.get)
    top = int(logits.argmax())
    return Prediction(item.id, name, model.name_or_path, key, probs=probs, latency_ms=latency,
                      input_tokens=ids.shape[1], output_tokens=0,
                      extra={"top_token_is_option_letter": top in lids[:n],
                             "letter_mass": float(torch.softmax(logits, -1)[lids[:n]].sum())})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local path")
    ap.add_argument("--name", help="results file name (default: derived from --model and --device)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--threads", type=int, help="CPU threads (torch.set_num_threads)")
    ap.add_argument("--limit", type=int, help="only the first N items (whole pairs)")
    ap.add_argument("--zero-shot", action="store_true", help="no worked examples")
    ap.add_argument("--temperature", type=float, default=1.0, help="calibration temperature for the option probabilities")
    ap.add_argument("--format", default="chat", choices=("chat", "compact"),
                    help="DecideBench's chat prompt, or Unee's compact prompt (what our models are trained on)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--load-4bit", action="store_true", help="load the model in 4-bit (big models on small GPUs)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    items = load_items()
    if problems := validate(items):
        sys.exit("dataset invalid:\n" + "\n".join(problems))
    if args.limit:
        keep = sorted({it.pair_id for it in items})[: (args.limit + 1) // 2]
        items = [it for it in items if it.pair_id in keep]

    dtype = {"auto": torch.bfloat16 if args.device == "cuda" else torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.load_4bit, dtype, args.device)
    lids = letter_ids(tok)
    name = args.name or (args.model.rstrip("/").split("/")[-1].lower() + f"-{args.device}"
                         + ("-zs" if args.zero_shot else ""))

    for it in items[: args.warmup]:
        predict(model, tok, lids, it, name, args.device, args.zero_shot, args.format, args.temperature)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.jsonl"
    t0 = time.perf_counter()
    with out.open("w") as f:
        for i, it in enumerate(items, 1):
            f.write(json.dumps(predict(model, tok, lids, it, name, args.device, args.zero_shot, args.format, args.temperature).to_json()) + "\n")
            if i % 50 == 0 or i == len(items):
                print(f"[{name}] {i}/{len(items)}  {i / (time.perf_counter() - t0):.1f} items/s", flush=True)

    rows = list(load_rows(out, {it.id: it for it in items}).values())
    s = summarize(name, rows, FREE)
    fam: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        fam[r.item.category].append(r.correct)
    letter_top = sum(r.extra["top_token_is_option_letter"] for r in rows) / len(rows)
    summary = {
        "name": name, "model": args.model, "device": args.device, "dtype": str(dtype).removeprefix("torch."),
        "threads": torch.get_num_threads() if args.device == "cpu" else None,
        "cpu": platform.processor(), "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "zero_shot": args.zero_shot, "format": args.format, "temperature": args.temperature, "n": s["n"], "acc": s["acc"], "ci": s["ci"], "pair_acc": s["pair_acc"],
        "ece": s["ece"], "brier": s["brier"], "p50_ms": s["p50"], "p95_ms": s["p95"], "avg_in_tokens": s["avg_in"],
        "top_token_is_option_letter": letter_top,
        "by_family": {k: sum(v) / len(v) for k, v in sorted(fam.items())},
    }
    (RESULTS / f"{name}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
