"""DecideBench through a Jev-compatible /v1/systemone server (ours), using DecideBench's own Jev adapter.

    python -m unee.server --llama http://127.0.0.1:8090 --port 8000
    PYTHONUTF8=1 .venv/Scripts/python bench/run_api.py --url http://127.0.0.1:8000/v1 --name unee-v0-api

This is how a leaderboard entry is scored: worked examples go inside each option's criteria entry, as TypeSafe
documents (`examples_in = "criteria"`), and the server builds its own prompt from them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "decidebench"))

import httpx  # noqa: E402

from decidebench.dataset import load_items  # noqa: E402
from decidebench.score import load_rows, summarize  # noqa: E402
from decidebench.systems.openjev import OpenJevSystem  # noqa: E402
from decidebench.types import FREE  # noqa: E402


class UneeSystem(OpenJevSystem):
    name = served_name = "unee"
    label = "Unee 0.8B"
    endpoint = "unee.server on llama.cpp"
    env_url, default_url = "UNEE_BASE_URL", "http://127.0.0.1:8000/v1"
    examples_in = "criteria"


async def run(url: str, name: str, concurrency: int) -> Path:
    import os
    os.environ["UNEE_BASE_URL"] = url
    system = UneeSystem()
    items = load_items()
    out = HERE / "results" / f"{name}.jsonl"
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        async def one(it):
            async with sem:
                return await system.predict(client, it)
        preds = await asyncio.gather(*(one(it) for it in items))
    out.write_text("".join(json.dumps(p.to_json()) + "\n" for p in preds))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--name", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()
    out = asyncio.run(run(args.url, args.name, args.concurrency))
    items = {it.id: it for it in load_items()}
    rows = list(load_rows(out, items).values())
    s = summarize(args.name, rows, FREE)
    fam: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        fam[r.item.category].append(r.correct)
    summary = {k: s[k] for k in ("n", "acc", "pair_acc", "ece", "p50", "p95", "unusable")}
    summary["by_family"] = {k: sum(v) / len(v) for k, v in sorted(fam.items())}
    (HERE / "results" / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
