"""Latency of Unee on a llama-server, in the situations that matter for an app.

    PYTHONUTF8=1 .venv/Scripts/python bench/speed.py --url http://127.0.0.1:8090/v1 --name unee-0.8b-cuda-q4

Scenarios (each uses real DecideBench inputs in Unee's compact prompt):
  few_shot     every request has a new question with one worked example per option (DecideBench's protocol)
  repeat       the same question and examples, with a new input each time: the usual production pattern,
               where the server reuses the cached prompt prefix
  zero_shot    no worked examples
Each is measured with 1 request in flight and with 4 (DecideBench measures self-hosted entries at 4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "decidebench"))
sys.path.insert(0, str(HERE.parent))

import httpx  # noqa: E402

from decidebench import fewshot  # noqa: E402
from decidebench.dataset import load_items  # noqa: E402
from unee.prompt import build_messages  # noqa: E402


def body(item, examples: bool, state=None) -> dict:
    opts = [{"key": o.key, "description": o.description} for o in item.options]
    shots = [(e.state, e.gold) for e in fewshot.for_item(item)] if examples else []
    return {"messages": build_messages(state if state is not None else item.state, item.question, opts, shots),
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}}


async def measure(url: str, bodies: list[dict], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    async with httpx.AsyncClient(timeout=600) as client:
        for b in bodies[:2]:  # warm-up
            await client.post(url, json=b)

        async def one(b):
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(url, json=b)
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(b) for b in bodies))
        wall = time.perf_counter() - t0
    lat.sort()
    return {"n": len(lat), "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "throughput_per_s": round(len(lat) / wall, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/chat/completions"
    items = load_items()[: args.n]
    # repeat: one question (the first item's) asked about many different inputs from the same family
    first = items[0]
    same_family = [it for it in load_items() if it.category == first.category][: args.n]
    scenarios = {
        "few_shot": [body(it, True) for it in items],
        "repeat": [body(first, True, state=it.state) for it in same_family],
        "zero_shot": [body(it, False) for it in items],
    }
    out = {"name": args.name, "url": args.url}
    for name, bodies in scenarios.items():
        for c in (1, 4):
            out[f"{name}_c{c}"] = asyncio.run(measure(url, bodies, c))
            print(name, c, out[f"{name}_c{c}"], flush=True)
    (HERE / "results" / f"{args.name}.speed.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
