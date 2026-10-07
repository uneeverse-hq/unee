"""Does fine-tuning keep the model's chat ability? Answer held-out user prompts, then let the teacher judge.

    # answers (GPU): writes <out>
    PYTHONUTF8=1 .venv/Scripts/python train/eval_chat.py answer --model Qwen/Qwen3.5-0.8B --out models/chat_base.json
    PYTHONUTF8=1 .venv/Scripts/python train/eval_chat.py answer --model models/v1/merged --out models/v1/chat.json
    # judge (teacher llama-server on :8080): blind pairwise, both orders
    PYTHONUTF8=1 .venv/Scripts/python train/eval_chat.py judge --a models/chat_base.json --b models/v1/chat.json

Prompts: oasst1 English first turns that chat_distill.py never used (index 1500 onward of chat_prompts.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
JUDGE = """Two assistants answered the same user message. Which answer is more helpful, correct and well written?
User message:
{prompt}

Answer 1:
{a}

Answer 2:
{b}

Reply with exactly one character: 1, 2, or = if they are equally good."""


def answer(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [json.loads(l)["prompt"] for l in (ROOT / "data/generated/chat_prompts.jsonl").read_text(
        encoding="utf-8").splitlines()[1500:1500 + args.n]]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False,
                                 eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])
        out.append({"prompt": p, "answer": tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()})
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} answers to {args.out}")


def judge(args) -> None:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    url = args.url.rstrip("/") + "/chat/completions"
    score = {"a": 0.0, "b": 0.0}
    with httpx.Client(timeout=600) as client:
        for x, y in zip(a, b):
            for first, second, names in ((x, y, ("a", "b")), (y, x, ("b", "a"))):
                msg = JUDGE.format(prompt=x["prompt"], a=first["answer"], b=second["answer"])
                r = client.post(url, json={"messages": [{"role": "user", "content": msg}], "max_tokens": 1,
                                           "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                verdict = r.json()["choices"][0]["message"]["content"].strip()[:1]
                if verdict == "1":
                    score[names[0]] += 1
                elif verdict == "2":
                    score[names[1]] += 1
                else:
                    score["a"] += 0.5
                    score["b"] += 0.5
    total = score["a"] + score["b"]
    result = {"a": args.a, "b": args.b, "judgements": total, "b_win_rate": round(score["b"] / total, 3)}
    print(json.dumps(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("answer")
    an.add_argument("--model", required=True)
    an.add_argument("--out", required=True)
    an.add_argument("--n", type=int, default=20)
    an.add_argument("--max-tokens", type=int, default=256)
    jd = sub.add_parser("judge")
    jd.add_argument("--a", required=True, help="reference answers (the base model)")
    jd.add_argument("--b", required=True, help="answers to compare (the fine-tuned model)")
    jd.add_argument("--url", default="http://127.0.0.1:8080/v1")
    jd.add_argument("--out")
    args = ap.parse_args()
    answer(args) if args.cmd == "answer" else judge(args)


if __name__ == "__main__":
    sys.exit(main())
