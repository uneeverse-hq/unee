"""Multilingual check on MASSIVE (Amazon; 51 languages, the same utterances written by native speakers in each).

    PYTHONUTF8=1 .venv/Scripts/python bench/run_massive.py --url http://127.0.0.1:8000 --name unee-0.8b --n 100

Task: which of 18 voice-assistant scenarios a request belongs to, through Unee's own /v1/systemone API (a choice
with 18 options, so the API's tournament is exercised too). The instruction and options stay in English while
the request is in each language, which is the usual shape for an app with users worldwide. The same `--n`
utterance ids are used in every language, so per-language scores compare like for like. Data: the MASSIVE test
split as packaged by MTEB (`mteb/amazon_massive_scenario`, Apache-2.0), files in bench/massive/<lang>.json.gz
(git-ignored). Fetch each language from PowerShell (HF downloads are flaky from Git Bash here):
    foreach ($l in "af am ar az bn cy da de el en es fa fi fr he hi hu hy id is it ja jv ka km kn ko lv ml mn ms my nb nl pl pt ro ru sl sq sv sw ta te th tl tr ur vi zh-CN zh-TW".Split(" ")) {
      Invoke-WebRequest "https://huggingface.co/datasets/mteb/amazon_massive_scenario/resolve/main/test/$l.json.gz" -OutFile "bench\\massive\\$l.json.gz" }
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import statistics
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
QUESTION = "Which part of the voice assistant should handle this request?"
SCENARIOS = {
    "alarm": "Set, check or cancel alarms.",
    "audio": "Change the device's volume or mute it.",
    "calendar": "Calendar events, meetings and reminders: create, check or remove them.",
    "cooking": "Recipes and how to cook something.",
    "datetime": "Ask the current date or time, or the time somewhere else.",
    "email": "Read, send or manage email and email contacts.",
    "general": "Small talk: jokes, greetings, questions about the assistant itself, or unclear requests.",
    "iot": "Control smart home devices: lights, plugs, vacuum cleaner, coffee machine, blinds.",
    "lists": "Create, read or change to-do and shopping lists.",
    "music": "Music preferences and settings: like or dislike a song, shuffle, ask what is playing.",
    "news": "Ask for news headlines or news on a topic.",
    "play": "Start playing something: music, radio, podcasts, audiobooks or games.",
    "qa": "Factual questions: definitions, facts, maths, currency or stock prices.",
    "recommendation": "Ask for suggestions: events, places, restaurants or movies.",
    "social": "Social media posts and complaints to companies.",
    "takeaway": "Order food for delivery or takeaway, or ask about an order.",
    "transport": "Travel: tickets, taxis, trains, traffic and directions.",
    "weather": "Weather conditions and forecasts.",
}


def load(lang: str) -> dict[str, dict]:
    with gzip.open(HERE / "massive" / f"{lang}.json.gz", "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


async def main_async(args) -> None:
    langs = sorted(p.name.split(".")[0] for p in (HERE / "massive").glob("*.json.gz"))
    if args.langs:
        langs = [l for l in langs if l in args.langs.split(",")]
    en = load("en")
    ids = sorted(en, key=int)
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]
    url = args.url.rstrip("/") + "/v1/systemone"
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, float] = {}
    async with httpx.AsyncClient(timeout=600) as client:
        for lang in langs:
            rows = load(lang)

            async def one(i: str) -> bool | None:
                row = rows.get(i)
                if row is None:
                    return None
                body = {"state": row["text"], "questions": {"s": {"type": "choice", "instructions": QUESTION,
                                                                   "criteria": SCENARIOS}}}
                async with sem:
                    r = await client.post(url, json=body)
                r.raise_for_status()
                return r.json()["answers"]["s"]["choice"] == row["label_text"]

            got = [g for g in await asyncio.gather(*(one(i) for i in ids)) if g is not None]
            results[lang] = sum(got) / len(got)
            print(f"{lang:6s} {results[lang]:.3f} ({len(got)})", flush=True)
    summary = {"name": args.name, "n": args.n, "seed": args.seed, "per_language": results,
               "mean": statistics.mean(results.values()), "english": results.get("en"),
               "languages_within_10pt_of_english": sum(v >= results.get("en", 0) - 0.10 for v in results.values())}
    out = HERE / "results" / f"massive-{args.name}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_language"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100, help="utterance ids per language (the same ids everywhere)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--langs", help="comma-separated subset, e.g. en,es,si (default: all 51)")
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
