"""Teacher-written chat data, so Unee chats like a bigger model. Three kinds:
- general: answers to real user prompts
- knowledge: answers grounded in a knowledge base, including "I don't know" when it lacks the answer
- summary: summaries of long multi-channel threads

    PYTHONUTF8=1 .venv/Scripts/python data/chat_teacher.py --hours 3

The teacher (llama-server on :8080) writes everything:
- **General prompts** are oasst1 English first turns 0-1499 of chat_prompts.jsonl. Prompts from 1500 onward are held
  out for train/eval_chat.py. The teacher is asked to be concise, but that request isn't stored, so the student
  learns short answers by default (shorter answers also stream faster).
- **Knowledge bases and threads** are invented by the teacher for the domains in families.py.
- **Knowledge questions** go through the same retrieval and prompt as serving (unee.knowledge.ground), so the
  student trains on exactly what it will see.

Jobs are interleaved, so a run cut short by --hours still gives a balanced mix. Output is resumable. Each row is
{"id", "kind", "messages", "response", "truncated"}.
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from families import DOMAINS  # noqa: E402
from gen import chat  # noqa: E402
from languages import pick  # noqa: E402
from unee.knowledge import ground  # noqa: E402
from unee.prompt import summary_messages  # noqa: E402

CONCISE = ("Give a helpful, accurate answer. Be concise: answer directly, and only add the detail the question needs. "
           "Use a short list only when it really helps.")
KB_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["company", "documents", "questions"],
    "properties": {
        "company": {"type": "string"},
        "documents": {"type": "array", "minItems": 3, "maxItems": 7, "items": {
            "type": "object", "additionalProperties": False, "required": ["title", "text"],
            "properties": {"title": {"type": "string"}, "text": {"type": "string"}}}},
        "questions": {"type": "array", "minItems": 4, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "required": ["question", "answerable"],
            "properties": {"question": {"type": "string"}, "answerable": {"type": "boolean"}}}},
    },
}
KB_PROMPT = """Invent a small {domain} business with an original name. Write the knowledge its customer chat \
assistant would use: {n_docs} short documents (FAQ, policies, prices, opening hours, product or service details, \
how-to steps), each 80-200 words, full of specific facts (numbers, dates, names, conditions, exceptions).

Then write 6 questions customers might type into the chat, in varied styles (short, messy with typos, polite, \
annoyed, two questions in one): 4 that the documents answer (at least one needing two facts combined) and 2 that \
sound related but that the documents do NOT answer. Mark each with "answerable"."""
THREAD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["thread"],
                 "properties": {"thread": {"type": "string"}}}
THREAD_STYLES = [
    "a customer's history across email, live chat and a phone-call note, in time order, each part labelled with "
    "its channel and date",
    "a long email chain between a customer and two staff members",
    "a WhatsApp-style chat between a customer and a support agent, with timestamps",
    "internal meeting notes about one customer account",
    "a support ticket with the customer's report, several agent replies and internal notes",
    "a transcript of a sales call with a prospective customer",
]
THREAD_PROMPT = """Write {style} for a {domain} business. 250-500 words. Use specific names, dates, amounts and \
product details, include some back-and-forth, at least one thing that gets resolved and at least one that is \
still open at the end. Write only the thread itself."""
ML_SHARE = 0.3  # share of jobs written in another language (data/languages.py), so Unee chats in the user's language
FOCUS = ["what the customer still needs", "deadlines and amounts", "next steps for the agent",
         "the customer's mood and any risk of losing them"]
REFUSAL = re.compile(r"don.t (know|have)|do not (know|have)|not sure|couldn.t find|could not find|no information|"
                     r"isn.t (in|covered|mentioned)|not (in|covered|mentioned|listed)|pass (this|your|the) question|"
                     r"(connect|put) you (with|in touch)|check with (a|our) (colleague|team|person)", re.I)


async def teacher(client, url: str, messages: list[dict], max_tokens: int, temperature: float,
                  schema: dict | None = None) -> tuple[str, bool]:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    c = (await chat(client, url, body))["choices"][0]
    return (c["message"].get("content") or "").strip(), c.get("finish_reason") != "stop"


async def says_unknown(client, url, text: str) -> bool:
    """For non-English replies, where the refusal regex cannot look: does the reply say the answer isn't known?"""
    verdict, _ = await teacher(client, url, [{"role": "user", "content": (
        "Does this customer-service reply say that the information is not available or that it does not know? "
        f"Reply yes or no.\n\nReply: {text}")}], 2, 0.0)
    return verdict.lower().startswith("y")


async def job_general(client, url, i, prompt, rng: random.Random) -> list[dict]:
    lang = pick(rng, ML_SHARE)
    if lang:  # the same real prompt, asked in another language
        prompt, cut = await teacher(client, url, [{"role": "user", "content": (
            f"Translate this message into {lang}. Reply with the translation only.\n\n{prompt}")}], 500, 0.2)
        if cut or not prompt:
            return []
    system = CONCISE + " Reply in the language of the user's message."
    text, cut = await teacher(client, url, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                              700, 0.7)
    return [{"id": f"g{i}", "kind": "general", "lang": lang or "English",
             "messages": [{"role": "user", "content": prompt}], "response": text, "truncated": cut}] if text else []


async def job_knowledge(client, url, i, rng: random.Random) -> list[dict]:
    lang = pick(rng, ML_SHARE)
    ask = KB_PROMPT.format(domain=rng.choice(DOMAINS), n_docs=rng.randint(4, 6))
    if lang:
        ask += f"\n\nWrite the company name, the documents and the questions in {lang}."
    raw, cut = await teacher(client, url, [{"role": "user", "content": ask}], 4000, 0.9, KB_SCHEMA)
    if cut:
        return []
    kb = json.loads(raw)
    docs = [d for d in kb["documents"] if d["text"].strip()]
    rows, history = [], []
    own = [{"role": "system", "content": f"You are the customer chat assistant for {kb['company']}."}] \
        if rng.random() < 0.5 else []
    for j, q in enumerate(kb["questions"]):
        # Sometimes keep the earlier turn, so the student also sees follow-up questions.
        prev = history[-2:] if history and rng.random() < 0.3 else []
        msgs = ground(own + prev + [{"role": "user", "content": q["question"]}], docs)
        text, cut = await teacher(client, url, msgs, 350, 0.3)
        if not text:
            continue
        if not q["answerable"] and not (await says_unknown(client, url, text) if lang else REFUSAL.search(text)):
            continue  # an unanswerable question answered anyway would teach making things up
        rows.append({"id": f"k{i}", "kind": "knowledge", "lang": lang or "English", "messages": msgs, "response": text,
                     "truncated": cut, "answerable": q["answerable"]})
        history = [{"role": "user", "content": q["question"]}, {"role": "assistant", "content": text}]
    return rows


async def job_summary(client, url, i, rng: random.Random) -> list[dict]:
    lang = pick(rng, ML_SHARE)
    ask = THREAD_PROMPT.format(style=rng.choice(THREAD_STYLES), domain=rng.choice(DOMAINS))
    if lang:
        ask += f" Write it in {lang}."
    thread_raw, cut = await teacher(client, url, [{"role": "user", "content": ask}], 2000, 0.9, THREAD_SCHEMA)
    if cut:
        return []
    msgs = summary_messages(json.loads(thread_raw)["thread"].strip(), rng.choice(FOCUS) if rng.random() < 0.3 else None)
    # The teacher tends to summarise in English whatever the text's language; this nudge is not stored, so the
    # stored example still matches the prompt the student sees ("in the language of the text").
    nudge = [{"role": "system", "content": f"Write the summary in {lang}."}] if lang else []
    text, cut = await teacher(client, url, nudge + msgs, 500, 0.3)
    return [{"id": f"s{i}", "kind": "summary", "lang": lang or "English", "messages": msgs, "response": text,
             "truncated": cut}] if text else []


async def main_async(args) -> None:
    prompts = [json.loads(l)["prompt"] for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip()][:1500]  # 1500 onward: held out for train/eval_chat.py
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if out.exists() else set()
    jobs = []  # interleaved: per 3 prompts, 3 general answers, 1 knowledge base (about 5 rows) and 1 summary
    for i in range(len(prompts)):
        kinds = [("g", i)] + ([("k", i)] if i % 3 == 1 else []) + ([("s", i)] if i % 3 == 0 else [])
        jobs += [(k, n) for k, n in kinds if f"{k}{n}" not in done]
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    deadline = time.time() + args.hours * 3600
    stats = {"general": 0, "knowledge": 0, "summary": 0, "failed": 0}
    print(f"{len(jobs)} jobs to run ({len(done)} ids done), stopping new jobs after {args.hours} h", flush=True)
    async with httpx.AsyncClient(timeout=900) as client:
        with out.open("a", encoding="utf-8") as f:
            async def one(kind: str, n: int) -> None:
                async with sem:
                    if time.time() > deadline:
                        return
                    rng = random.Random(f"{args.seed}-{kind}{n}")
                    try:
                        rows = await (job_general(client, url, n, prompts[n], rng) if kind == "g" else
                                      job_knowledge(client, url, n, rng) if kind == "k" else
                                      job_summary(client, url, n, rng))
                    except Exception as e:  # noqa: BLE001 - one bad job is skipped, never fatal
                        stats["failed"] += 1
                        print(f"{kind}{n} failed: {type(e).__name__}: {e}"[:200], flush=True)
                        return
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    stats[r["kind"]] += 1
                f.flush()
                if sum(stats.values()) % 20 < len(rows) or not rows:
                    print(f"{stats}", flush=True)

            await asyncio.gather(*(one(k, n) for k, n in jobs))
    print(f"done: {stats}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--prompts", default=str(HERE / "generated" / "chat_prompts.jsonl"))
    ap.add_argument("--out", default=str(HERE / "generated" / "chat_teacher.jsonl"))
    ap.add_argument("--hours", type=float, default=3.0, help="no new jobs start after this long")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
