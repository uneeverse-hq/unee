"""Run a model served by an OpenAI-compatible server (llama-server) on DecideBench, scoring options from logprobs.

    tools/llama.cpp/cuda/llama-server.exe -m models/gguf/Qwen3.5-9B-Q4_K_M.gguf --jinja -ngl 99 -np 4 -c 32768 --port 8080
    PYTHONUTF8=1 .venv/Scripts/python bench/run_server.py --name qwen35-9b-q4 --concurrency 4

Same prompt as bench/run_local.py (DecideBench's chat prompt with one worked example per option). The server returns
the top logprobs at the first answer position; each option's probability is its letter's share among them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "decidebench"))
sys.path.insert(0, str(HERE.parent))

import httpx  # noqa: E402

from decidebench import fewshot  # noqa: E402
from decidebench.dataset import Item, load_items, validate  # noqa: E402
from decidebench.prompts import LETTERS, build_messages  # noqa: E402
from unee.prompt import build_messages as compact_messages  # noqa: E402
from decidebench.score import load_rows, summarize  # noqa: E402
from decidebench.types import FREE, Prediction  # noqa: E402

RESULTS = HERE / "results"


def item_messages(item: Item, zero_shot: bool, fmt: str, reverse: bool = False) -> list[dict]:
    """DecideBench's chat prompt, or Unee's compact one; both get the item's seeded worked examples.
    `reverse` lists the options last-to-first (compact format only)."""
    shots = [] if zero_shot else fewshot.for_item(item)
    if fmt == "compact":
        opts = [{"key": o.key, "description": o.description} for o in item.options]
        return compact_messages(item.state, item.question, opts[::-1] if reverse else opts,
                                [(e.state, e.gold) for e in shots])
    return build_messages(item, shots)


def letter_probs(item: Item, top: list[dict], reverse: bool = False) -> dict[str, float]:
    """Option probabilities from the top logprobs, renormalised over the option letters; unseen letters get a floor."""
    seen: dict[str, float] = {}
    for t in top:  # "A" and " A" are different tokens; both count for option A
        seen[t["token"].strip()] = seen.get(t["token"].strip(), 0.0) + math.exp(t["logprob"])
    floor = min(seen.values(), default=1e-6) * 0.5
    opts = item.options[::-1] if reverse else item.options
    raw = {o.key: seen.get(LETTERS[i], floor) for i, o in enumerate(opts)}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


async def predict(client: httpx.AsyncClient, url: str, item: Item, name: str, zero_shot: bool, fmt: str,
                  orders: int = 1) -> Prediction:
    """With orders=2 the item is also asked with the options reversed, and the two distributions are averaged."""
    runs = []
    t0 = time.perf_counter()
    for reverse in (False, True)[:orders]:
        body = {"messages": item_messages(item, zero_shot, fmt, reverse), "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        runs.append(letter_probs(item, top, reverse))
    latency = (time.perf_counter() - t0) * 1000
    probs = {k: sum(p[k] for p in runs) / len(runs) for k in runs[0]}
    if orders > 1:
        top = [{"token": LETTERS[0]}]  # the top-token check is only meaningful for a single order
    letters = set(LETTERS[: len(item.options)])
    return Prediction(item.id, name, data.get("model", ""), max(probs, key=probs.get), probs=probs,
                      latency_ms=latency, input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                      extra={"top_token_is_option_letter": top[0]["token"].strip() in letters})


async def run(args, items: list[Item], out: Path) -> None:
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        for it in items[: args.warmup]:
            await predict(client, url, it, args.name, args.zero_shot, args.format, args.orders)
        done = 0
        t0 = time.perf_counter()
        with out.open("w") as f:
            async def one(it: Item) -> None:
                nonlocal done
                async with sem:
                    p = await predict(client, url, it, args.name, args.zero_shot, args.format, args.orders)
                f.write(json.dumps(p.to_json()) + "\n")
                done += 1
                if done % 50 == 0 or done == len(items):
                    print(f"[{args.name}] {done}/{len(items)}  {done / (time.perf_counter() - t0):.1f} items/s", flush=True)
            await asyncio.gather(*(one(it) for it in items))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--format", default="chat", choices=("chat", "compact"))
    ap.add_argument("--orders", type=int, default=1, choices=(1, 2), help="2: also ask with options reversed, average")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    items = load_items()
    if problems := validate(items):
        sys.exit("dataset invalid:\n" + "\n".join(problems))
    if args.limit:
        keep = sorted({it.pair_id for it in items})[: (args.limit + 1) // 2]
        items = [it for it in items if it.pair_id in keep]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.jsonl"
    asyncio.run(run(args, items, out))

    rows = list(load_rows(out, {it.id: it for it in items}).values())
    s = summarize(args.name, rows, FREE)
    fam: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        fam[r.item.category].append(r.correct)
    summary = {"name": args.name, "server": args.url, "concurrency": args.concurrency, "zero_shot": args.zero_shot, "format": args.format, "orders": args.orders,
               "n": s["n"], "acc": s["acc"], "ci": s["ci"], "pair_acc": s["pair_acc"], "ece": s["ece"],
               "brier": s["brier"], "p50_ms": s["p50"], "p95_ms": s["p95"], "avg_in_tokens": s["avg_in"],
               "top_token_is_option_letter": sum(r.extra["top_token_is_option_letter"] for r in rows) / len(rows),
               "by_family": {k: sum(v) / len(v) for k, v in sorted(fam.items())}}
    (RESULTS / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
