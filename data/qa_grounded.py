"""Grounded-answer training rows from SQuAD 2.0 (CC BY-SA 4.0): answer from the passages, or say you don't know.

    PYTHONUTF8=1 .venv/Scripts/python data/qa_grounded.py --n 2000 --out data/generated/qa_grounded.jsonl

Each row is the exact serving prompt (unee.prompt.knowledge_messages): the SQuAD paragraph plus three paragraphs
from other articles, shuffled, the way retrieval hands the website assistant several passages.
- **Half the questions are SQuAD 2.0's unanswerable ones.** They are written to look answerable from their
  paragraph, which is what teaches a small model not to guess.
- **About 30% of questions are translated** into another language (data/languages.py weights; Arabic, Hindi and the
  other top languages first). The passages stay in English: an English website answering visitors in their own
  language.

The teacher (Qwen3.5-9B on :8080) writes the reply with a private hint (the verified answer span, or that the
passages don't answer it). Rows are kept only if:
- an answerable reply states the span (or its numbers, when translated)
- an unanswerable reply declines (regex for English, teacher check otherwise)

The hint is never stored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import httpx
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from chat_teacher import REFUSAL, says_unknown, teacher  # noqa: E402
from languages import pick  # noqa: E402

from unee.prompt import knowledge_messages  # noqa: E402

ML_SHARE = 0.3
HINT_YES = ("(Private note, not visible to the user: the answer is \"{span}\". Reply in {lang}. Write one or two short, "
            "friendly sentences that give this answer, using only facts in the knowledge. Do not mention passages or "
            "this note.)")
HINT_NO = ("(Private note, not visible to the user: the knowledge does not answer this question. Reply in {lang}. Say "
           "briefly that you don't have that information and offer to pass the question to a person. Do not guess, "
           "and do not mention passages or this note.)")


async def job(client, url: str, i: int, ex: dict, distractors: list[str], rng: random.Random) -> dict | None:
    lang = pick(rng, ML_SHARE)
    question = ex["question"].strip()
    if lang:
        question, cut = await teacher(client, url, [{"role": "user", "content": (
            f"Translate this question into {lang}. Keep names and numbers as they are. Reply with the translation "
            f"only.\n\n{question}")}], 200, 0.2)
        if cut or not question:
            return None
    passages = [ex["context"].strip()] + distractors
    rng.shuffle(passages)
    msgs = knowledge_messages([{"role": "user", "content": question}], passages)
    answers = [a for a in ex["answers"]["text"] if a.strip()]
    answerable = bool(answers)
    hint = (HINT_YES.format(span=answers[0], lang=lang or "English") if answerable
            else HINT_NO.format(lang=lang or "English"))
    text, cut = await teacher(client, url, msgs[:-1] + [{"role": "user", "content": f"{question}\n\n{hint}"}],
                              160, 0.5)
    if cut or not text:
        return None
    if answerable:
        span = answers[0]
        digits = re.findall(r"\d+", span)
        ok = span.lower() in text.lower() if not lang else all(d in text for d in digits)
        if not ok:
            return None
    elif not (await says_unknown(client, url, text) if lang else REFUSAL.search(text)):
        return None
    return {"id": f"qa{i}", "kind": "knowledge", "lang": lang or "English", "messages": msgs, "response": text,
            "truncated": False, "answerable": answerable, "src": "squad_v2"}


async def main_async(args) -> None:
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    rng = random.Random(args.seed)
    yes = [i for i in range(len(ds)) if ds[i]["answers"]["text"]]
    no = [i for i in range(len(ds)) if not ds[i]["answers"]["text"]]
    picks = rng.sample(yes, args.n // 2) + rng.sample(no, args.n // 2)
    rng.shuffle(picks)
    contexts = sorted(set(ds["context"]))
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if out.exists() else set()
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"kept": 0, "dropped": 0}
    async with httpx.AsyncClient(timeout=600) as client:
        with out.open("a", encoding="utf-8") as f:
            async def one(n: int, idx: int) -> None:
                if f"qa{n}" in done:
                    return
                ex = ds[idx]
                r = random.Random(f"{args.seed}-{n}")
                distractors = [c for c in r.sample(contexts, 6) if c != ex["context"]][:3]
                async with sem:
                    try:
                        row = await job(client, url, n, ex, distractors, r)
                    except Exception as e:  # noqa: BLE001
                        print(f"qa{n} failed: {type(e).__name__}"[:120], flush=True)
                        row = None
                if row:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                stats["kept" if row else "dropped"] += 1
                if sum(stats.values()) % 50 == 0:
                    print(stats, flush=True)

            await asyncio.gather(*(one(n, idx) for n, idx in enumerate(picks)))
    print("done", stats, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "generated" / "qa_grounded.jsonl"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
