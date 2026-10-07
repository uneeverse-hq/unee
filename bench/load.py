"""Latency and throughput of the Unee API (/v1/systemone) under load, on DecideBench inputs with worked examples.

    PYTHONUTF8=1 .venv/Scripts/python bench/load.py --url http://127.0.0.1:8000 --name torch-0.8b

Each request is one choice question (DecideBench's item and its one-example-per-option criteria, as the benchmark's
own Jev adapter sends them). Measured with 1, 4 and 16 requests in flight.
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

import httpx  # noqa: E402

from decidebench import fewshot  # noqa: E402
from decidebench.dataset import load_items  # noqa: E402
from decidebench.systems.jev import build_body  # noqa: E402


async def run(url: str, bodies: list[dict], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    async with httpx.AsyncClient(timeout=600) as client:
        for b in bodies[:3]:
            (await client.post(url, json=b)).raise_for_status()

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
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=160)
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/v1/systemone"
    bodies = [build_body(it, "unee", fewshot.for_item(it), "criteria") for it in load_items()[: args.n]]
    out = {"name": args.name}
    for c in (1, 4, 16):
        out[f"c{c}"] = asyncio.run(run(url, bodies, c))
        print(c, out[f"c{c}"], flush=True)
    (HERE / "results" / f"{args.name}.load.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
