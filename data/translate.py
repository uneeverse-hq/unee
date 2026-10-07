"""Multilingual decision data: the teacher translates training items into other languages (data/languages.py).

    PYTHONUTF8=1 .venv/Scripts/python data/translate.py --inp data/generated/train.jsonl \
        --out data/generated/train_ml.jsonl --hours 1.5
    PYTHONUTF8=1 .venv/Scripts/python data/translate.py --inp data/generated/val.jsonl \
        --out data/generated/val_ml.jsonl --n 240 --seed 1 --hours 0.5

What gets translated:
- **Always:** the item's input (state) and its worked examples. Labels carry over, because a faithful translation
  keeps the answer.
- **In 40% of items:** the question and option descriptions too. The rest keep English instructions over
  foreign-language input, which is the common case for an app serving customers worldwide.

Rules:
- Keys never change.
- Names, numbers, dates, amounts and codes must survive verbatim, or the item is dropped. Contrastive pairs often
  hinge on exactly those details.
- The English "why" is dropped, so explanations stay English-sourced.
- Output is resumable. Each row is the input item plus "lang".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen import chat  # noqa: E402
from languages import pick  # noqa: E402

FULL_SHARE = 0.4
DIGITS = re.compile(r"\d+")
PROMPT = """Translate every text in this JSON into {lang}. Keep names, numbers, times, amounts, codes, email \
addresses and URLs exactly as they are, written with the same digits. Write dates the way a native {lang} speaker \
would (local month names and order), keeping their day and year numbers. Keep the meaning, tone and level of \
formality, and let the wording sound natural for a native {lang} speaker. Return the same JSON structure with the \
translations.

{payload}"""


def schema(n_examples: int, n_options: int) -> dict:
    props = {"state": {"type": "string"},
             "examples": {"type": "array", "minItems": n_examples, "maxItems": n_examples, "items": {"type": "string"}}}
    if n_options:
        props["question"] = {"type": "string"}
        props["options"] = {"type": "array", "minItems": n_options, "maxItems": n_options,
                            "items": {"type": "string"}}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def keeps_numbers(src: str, dst: str) -> bool:
    """Every digit run in the source appears in the translation (dates, amounts, IDs carry the decision)."""
    have = DIGITS.findall(dst)
    return all(d in have for d in DIGITS.findall(src))


async def translate(client, url: str, item: dict, lang: str, full: bool) -> dict | None:
    payload = {"state": item["state"], "examples": [e["state"] for e in item["examples"]]}
    if full:
        payload["question"] = item["question"]
        payload["options"] = [o["description"] for o in item["options"]]
    body = {"messages": [{"role": "user", "content": PROMPT.format(lang=lang, payload=json.dumps(payload, ensure_ascii=False))}],
            "max_tokens": 3000, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema(
                len(payload["examples"]), len(payload.get("options", [])))}}}
    c = (await chat(client, url, body))["choices"][0]
    if c.get("finish_reason") != "stop":
        return None
    t = json.loads(c["message"]["content"])
    pairs = [(payload["state"], t["state"]), *zip(payload["examples"], t["examples"])]
    if full:
        pairs += [(payload["question"], t["question"]), *zip(payload["options"], t["options"])]
    if any(not dst.strip() or not keeps_numbers(src, dst) for src, dst in pairs):
        return None
    out = {k: v for k, v in item.items() if k != "why"}
    out["state"] = t["state"]
    out["examples"] = [{**e, "state": s} for e, s in zip(item["examples"], t["examples"])]
    if full:
        out["question"] = t["question"]
        out["options"] = [{**o, "description": d} for o, d in zip(item["options"], t["options"])]
    out["lang"] = lang
    out["lang_scope"] = "full" if full else "input"
    return out


async def main_async(args) -> None:
    items = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    order = order[: args.n] if args.n else order
    out = Path(args.out)
    done = {json.loads(l)["ml_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if out.exists() else set()
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    deadline = time.time() + args.hours * 3600
    stats = {"ok": 0, "dropped": 0, "failed": 0}
    todo = [i for i in order if f"{args.seed}-{i}" not in done]
    print(f"{len(todo)} items to translate ({len(done)} done), stopping new ones after {args.hours} h", flush=True)
    async with httpx.AsyncClient(timeout=900) as client:
        with out.open("a", encoding="utf-8") as f:
            async def one(i: int) -> None:
                async with sem:
                    if time.time() > deadline:
                        return
                    r = random.Random(f"{args.seed}-{i}")
                    lang = pick(r)
                    try:
                        row = await translate(client, url, items[i], lang, r.random() < FULL_SHARE)
                    except Exception as e:  # noqa: BLE001 - one bad item is skipped, never fatal
                        stats["failed"] += 1
                        print(f"{i} failed: {type(e).__name__}: {e}"[:160], flush=True)
                        return
                if row is None:
                    stats["dropped"] += 1
                else:
                    row["ml_id"] = f"{args.seed}-{i}"
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    stats["ok"] += 1
                if (stats["ok"] + stats["dropped"]) % 25 == 0:
                    print(stats, flush=True)

            await asyncio.gather(*(one(i) for i in todo))
    print(f"done: {stats}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--n", type=int, default=0, help="translate at most this many items (0 = as many as time allows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=1.5, help="no new items start after this long")
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
