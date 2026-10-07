"""Does Unee answer from your documents, and say so when they don't cover the question? (The website-assistant case.)

    # 1. held-out knowledge bases (teacher on :8080): new companies, about 30% in other languages
    PYTHONUTF8=1 .venv/Scripts/python train/eval_knowledge.py make --n 24
    # 2. answers from any OpenAI-compatible server running the model under test (for example `unee serve`)
    PYTHONUTF8=1 .venv/Scripts/python train/eval_knowledge.py answer --url http://127.0.0.1:8021 --name unee-r3-0.8b
    # 3. teacher judgement (teacher on :8080)
    PYTHONUTF8=1 .venv/Scripts/python train/eval_knowledge.py judge --name unee-r3-0.8b

Strict mode (unee/strict.py): `answer --strict` asks the server to check each answer before sending it and saves
what the check did. `rebuild --name <that run> --out <new name> --threshold T` writes the replies the same drafts
would give at another threshold (negative: no check), so the before and after are judged on identical drafts.

The questions go through exactly the serving prompt (unee.knowledge.ground), so they're the requests the website
assistant sends.
- **Answerable questions** count as correct when the answer is right and adds nothing the knowledge doesn't say.
- **Unanswerable questions** count as correct when the answer declines.
- **Made-up facts** are judged separately ("invents"): does the reply state anything the knowledge does not contain?
  A reply that only declines is wrong for an answerable question, but it invents nothing.

The knowledge bases come from data/chat_teacher.py's generator with a seed and ids never used for training.
Results go to bench/results/knowledge-<name>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))

import chat_teacher  # noqa: E402

SET = ROOT / "data/generated/kb_eval.jsonl"
TEACHER = "http://127.0.0.1:8080/v1/chat/completions"
JUDGE = """You are checking a customer-service assistant that must answer only from the knowledge it was given.

{conversation}

Assistant's reply:
{reply}

Reference reply written by an expert with the same knowledge:
{reference}

The question {status}. Is the assistant's reply acceptable? For a question the knowledge answers, it must be \
correct and must not add facts the knowledge does not contain. For a question the knowledge does not answer, it \
must say it doesn't know or can't find it, rather than invent an answer. Reply with exactly one word: yes or no."""
# The made-up-facts question, separate from "acceptable": a reply that only declines is not acceptable for a question
# the knowledge answers, but it invents nothing.
INVENTS = """You are checking a customer-service assistant that must answer only from the knowledge it was given.

{conversation}

Assistant's reply:
{reply}

Does the assistant's reply state any fact that the knowledge does not contain, for example an invented number, \
name, price, policy or instruction? Saying it doesn't know, asking the user a question and offering to pass the \
question to a person are not facts. Reply with exactly one word: yes or no."""


async def make(args) -> None:
    rows = []
    async with httpx.AsyncClient(timeout=900) as client:
        sem = asyncio.Semaphore(4)

        async def one(i: int) -> None:
            async with sem:
                try:  # ids "kbeval-*" and this seed never appear in training data
                    got = await chat_teacher.job_knowledge(client, TEACHER, f"eval{i}", random.Random(f"kbeval-{i}"))
                except Exception as e:  # noqa: BLE001
                    print(f"{i} failed: {e}"[:120], flush=True)
                    return
            rows.extend(got)

        await asyncio.gather(*(one(i) for i in range(args.n)))
    SET.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} questions from {args.n} knowledge bases;",
          dict(collections.Counter((r["lang"], r["answerable"]) for r in rows).most_common(8)))


async def answer(args) -> None:
    rows = [json.loads(l) for l in SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=600) as client:
        async def one(r: dict) -> tuple[str, dict | None]:
            body = {"messages": r["messages"], "max_tokens": 350, "temperature": 0}
            if args.strict:
                body["strict"] = {"threshold": args.strict}
            async with sem:
                resp = await client.post(args.url.rstrip("/") + "/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"].get("content") or "").strip(), data.get("unee")

        got = await asyncio.gather(*(one(r) for r in rows))
    out = ROOT / f"models/kb_answers-{args.name}.json"
    out.write_text(json.dumps([text for text, _ in got], ensure_ascii=False, indent=1), encoding="utf-8")
    if args.strict:  # what strict mode did to each reply, for reading the failures
        (ROOT / f"models/kb_strict-{args.name}.json").write_text(
            json.dumps([report for _, report in got], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(got)} replies to {out}")


def rebuild(args) -> None:
    """Replies at another threshold, or with no check at all, from a strict run's saved reports. The drafts are the
    same, so runs judged this way compare like for like."""
    from unee.strict import assemble

    reports = json.loads((ROOT / f"models/kb_strict-{args.name}.json").read_text(encoding="utf-8"))
    threshold = None if args.threshold < 0 else args.threshold
    replies = [assemble(x["sentences"], None, threshold)[0] for x in reports]
    out = ROOT / f"models/kb_answers-{args.out}.json"
    out.write_text(json.dumps(replies, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(replies)} replies to {out}")


async def judge(args) -> None:
    rows = [json.loads(l) for l in SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    replies = json.loads((ROOT / f"models/kb_answers-{args.name}.json").read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=600) as client:
        async def ask(prompt: str) -> bool:
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 2, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with sem:
                resp = await client.post(TEACHER, json=body)
            return (resp.json()["choices"][0]["message"].get("content") or "").strip().lower().startswith("y")

        async def one(r: dict, reply: str) -> tuple[bool, bool]:
            convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in r["messages"])
            status = "is answered by the knowledge" if r["answerable"] else "is NOT answered by the knowledge"
            return await asyncio.gather(
                ask(JUDGE.format(conversation=convo, reply=reply, reference=r["response"], status=status)),
                ask(INVENTS.format(conversation=convo, reply=reply)))

        both = await asyncio.gather(*(one(r, a) for r, a in zip(rows, replies)))
    ok, invents = [b[0] for b in both], [b[1] for b in both]

    def rate(pick, flags=ok) -> float | None:
        got = [o for r, o in zip(rows, flags) if pick(r)]
        return round(sum(got) / len(got), 3) if got else None

    langs = sorted({r["lang"] for r in rows})
    out = {"name": args.name, "n": len(rows), "acceptable": rate(lambda r: True),
           "answerable": rate(lambda r: r["answerable"]), "unanswerable": rate(lambda r: not r["answerable"]),
           "english": rate(lambda r: r["lang"] == "English"), "other_languages": rate(lambda r: r["lang"] != "English"),
           "by_language": {l: rate(lambda r, l=l: r["lang"] == l) for l in langs},
           "invents": rate(lambda r: True, invents), "invents_answerable": rate(lambda r: r["answerable"], invents),
           "invents_unanswerable": rate(lambda r: not r["answerable"], invents),
           "per_question": [{"acceptable": o, "invents": i} for o, i in zip(ok, invents)]}
    (ROOT / f"bench/results/knowledge-{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_language", "per_question")}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--n", type=int, default=24)
    an = sub.add_parser("answer")
    an.add_argument("--url", required=True)
    an.add_argument("--name", required=True)
    an.add_argument("--strict", nargs="?", const=0.5, type=float, metavar="THRESHOLD",
                    help="ask for strict mode (unee/strict.py), optionally with its threshold")
    rb = sub.add_parser("rebuild")
    rb.add_argument("--name", required=True, help="a run made with `answer --strict`")
    rb.add_argument("--out", required=True, help="name for the rebuilt replies")
    rb.add_argument("--threshold", type=float, required=True, help="strict threshold; negative for no check")
    jd = sub.add_parser("judge")
    jd.add_argument("--name", required=True)
    args = ap.parse_args()
    if args.cmd == "rebuild":
        return rebuild(args)
    asyncio.run({"make": make, "answer": answer, "judge": judge}[args.cmd](args))


if __name__ == "__main__":
    main()
