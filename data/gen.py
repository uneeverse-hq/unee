"""Synthetic decision data from an open-weight teacher served by llama-server (OpenAI-compatible API).

    # 1. teacher writes templates: question, options, one worked example per option, contrastive pairs
    PYTHONUTF8=1 .venv/Scripts/python data/gen.py generate --n 200 --out data/generated/raw.jsonl
    # 2. teacher answers every pair item in the eval prompt format; its letter probabilities are kept
    PYTHONUTF8=1 .venv/Scripts/python data/gen.py label --inp data/generated/raw.jsonl --out data/generated/labeled.jsonl
    # 3. filter (teacher agrees with the designed answer, no DecideBench overlap) and split train/val by template
    PYTHONUTF8=1 .venv/Scripts/python data/gen.py build --inp data/generated/labeled.jsonl --out data/generated

Training never reads DecideBench. `build` reads it only to drop anything too close to a test item or example.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "bench" / "decidebench"))

sys.path.insert(0, str(ROOT))

from families import DOMAINS, EDITS, FAMILIES, STYLES  # noqa: E402
from unee.prompt import LETTERS, chat_messages  # noqa: E402

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "minItems": 2, "maxItems": 26, "items": {
            "type": "object", "properties": {"key": {"type": "string"}, "description": {"type": "string"}},
            "required": ["key", "description"]}},
        "examples": {"type": "array", "minItems": 2, "maxItems": 26, "items": {
            "type": "object", "properties": {"input": {"type": "string"}, "gold": {"type": "string"}},
            "required": ["input", "gold"]}},
        "pairs": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "properties": {
                "edit": {"type": "string"},
                "input_a": {"type": "string"}, "gold_a": {"type": "string"}, "why_a": {"type": "string"},
                "input_b": {"type": "string"}, "gold_b": {"type": "string"}, "why_b": {"type": "string"}},
            "required": ["edit", "input_a", "gold_a", "why_a", "input_b", "gold_b", "why_b"]}},
    },
    "required": ["question", "options", "examples", "pairs"],
}

PROMPT = """You write training data for a decision model. A decision model reads an input, a question and a list of \
options, and picks exactly one option.

Create ONE decision task.
- Family: {family}. Decide {what}.
- Domain: {domain}.
- Inputs are {style}.
- Options: {hint}. Use exactly {n} options. {kind_rule}

Write:
1. "question": one clear sentence of at most 60 words. If the decision depends on a written policy or rubric, state \
it briefly in the question, or put it at the start of every input.
2. "options": {n} options. Each has a snake_case "key" and a one-sentence "description" that says exactly when it \
applies. Options must not overlap, and together they must cover every case.
3. "examples": exactly one solved example per option, in the same order as the options ({n} in total). "gold" is \
that option's key.
4. "pairs": {k} contrastive pairs. "input_b" copies "input_a" and changes ONE small, realistic detail so that the \
correct answer changes. Use these kinds of edits, one per pair: {edits}. "gold_a" and "gold_b" must differ. \
"why_a" and "why_b" are one sentence each that explain the answer by pointing at the deciding detail.

Rules:
- Every input is 40-600 characters, specific and realistic, with names, products, numbers, dates or times. No \
placeholders like [NAME].
- A careful human reader must agree with every gold answer. If a detail would make an answer debatable, change it.
- Every gold answer must follow from the question and the option descriptions alone. Never rely on a rule, limit or fact that they do not state.
- Pair inputs are new: never copy or lightly edit a worked example.
- Spread the gold answers across all options. The first option must not be the usual answer.
- Keep each edit subtle: a reader who only matches topic words should get one half of the pair wrong.
- Vary names, writing style and length. Never reuse a person's or company's name within this task.
Return only the JSON object."""

KIND_RULES = {
    "choice": "",
    "noul": 'This is a yes/no decision: use exactly the keys "true" and "false", and phrase the question as a '
            "yes/no question.",
    "score": "The options are ordered levels from lowest to highest, in that order.",
}


def spec_for(i: int, seed: int, focus: frozenset = frozenset(), focus_weight: float = 3.0) -> dict:
    """The i-th template spec; families in `focus` are drawn `focus_weight` times as often."""
    rng = random.Random(f"{seed}-{i}")
    if focus:
        family, kind, what, hint = rng.choices(FAMILIES, [focus_weight if f[0] in focus else 1.0 for f in FAMILIES])[0]
    else:
        family, kind, what, hint = rng.choice(FAMILIES)
    listed = re.match(r"^[a-z_]+( / [a-z_]+)+$", hint.split(",")[0].strip())
    span = re.search(r"(\d+)-(\d+)", hint)
    if kind == "noul":
        n = 2
    elif listed:
        n = listed.group(0).count(" / ") + 1
    elif span:
        n = rng.randint(int(span.group(1)), int(span.group(2)))
    else:
        n = rng.choice([4, 5]) if kind == "score" else rng.choice([3, 4, 5])
    k = rng.choice([4, 5, 6])
    return {"tid": f"s{seed}-{i:06d}", "family": family, "kind": kind, "what": what, "hint": hint, "n": n, "k": k,
            "domain": rng.choice(DOMAINS), "style": rng.choice(STYLES), "edits": rng.sample(EDITS, k)}


def prompt_for(spec: dict) -> str:
    return PROMPT.format(family=spec["family"].replace("_", " "), what=spec["what"], domain=spec["domain"],
                         style=spec["style"], hint=spec["hint"], n=spec["n"], k=spec["k"],
                         kind_rule=KIND_RULES[spec["kind"]], edits="; ".join(spec["edits"]))


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def validate_template(spec: dict, t: dict) -> tuple[dict | None, list[str]]:
    """Clean a teacher template; returns (template or None, problems). Bad pairs are dropped, not fatal."""
    probs: list[str] = []
    keys = [o["key"].strip() for o in t["options"]]
    if len(keys) != spec["n"] or len(set(keys)) != len(keys) or not all(KEY_RE.match(k) for k in keys):
        return None, [f"bad option keys {keys}"]
    if spec["kind"] == "noul" and sorted(keys) != ["false", "true"]:
        return None, [f"noul keys {keys}"]
    ex = {e["gold"].strip(): e["input"].strip() for e in t["examples"]}
    if sorted(ex) != sorted(keys) or not all(30 <= len(v) <= 700 for v in ex.values()):
        return None, ["examples do not cover each option once"]
    pairs = []
    for p in t["pairs"]:
        a, b, ga, gb = p["input_a"].strip(), p["input_b"].strip(), p["gold_a"].strip(), p["gold_b"].strip()
        why = f"pair {len(pairs)}"
        if ga not in keys or gb not in keys or ga == gb:
            probs.append(f"{why}: golds {ga}/{gb}")
        elif not (30 <= len(a) <= 700 and 30 <= len(b) <= 700) or a == b:
            probs.append(f"{why}: input length or identical")
        elif similar(a, b) < 0.35:
            probs.append(f"{why}: edit too large ({similar(a, b):.2f})")
        elif any(similar(x, e) > 0.9 for x in (a, b) for e in ex.values()):
            probs.append(f"{why}: copies an example")
        else:
            pairs.append({"edit": p["edit"].strip(), "a": a, "gold_a": ga, "why_a": p["why_a"].strip(),
                          "b": b, "gold_b": gb, "why_b": p["why_b"].strip()})
    if not pairs:
        return None, probs + ["no valid pairs"]
    out = {k: spec[k] for k in ("tid", "family", "kind", "domain", "style")}
    out |= {"question": t["question"].strip(),
            "options": [{"key": o["key"].strip(), "description": o["description"].strip()} for o in t["options"]],
            "examples": [{"state": ex[k], "gold": k} for k in keys], "pairs": pairs}
    return out, probs


async def chat(client: httpx.AsyncClient, url: str, body: dict, attempts: int = 4) -> dict:
    for i in range(attempts):
        try:
            r = await client.post(url, json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            if i == attempts - 1:
                raise
            await asyncio.sleep(2 ** i)
    raise AssertionError


def done_tids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["tid"] for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()}


async def generate(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects = out.with_suffix(".rejects.jsonl")
    seen = done_tids(out) | done_tids(rejects)
    focus = frozenset(f for f in args.focus.split(",") if f)
    unknown = focus - {f[0] for f in FAMILIES}
    if unknown:
        raise SystemExit(f"unknown families in --focus: {sorted(unknown)}")
    specs = [s for s in (spec_for(i, args.seed, focus, args.focus_weight) for i in range(args.start, args.start + args.n))
             if s["tid"] not in seen]
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"ok": 0, "bad": 0, "pairs": 0, "tokens": 0}
    t0 = time.perf_counter()
    print(f"generate: {len(specs)} templates to write ({len(seen)} already done)", flush=True)
    async with httpx.AsyncClient(timeout=900) as client:
        with out.open("a", encoding="utf-8") as fo, rejects.open("a", encoding="utf-8") as fr:
            async def one(spec: dict) -> None:
                body = {"messages": [{"role": "user", "content": prompt_for(spec)}],
                        "temperature": args.temperature, "top_p": 0.95, "max_tokens": 4000 if spec["n"] <= 8 else 7000,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {"type": "json_schema", "json_schema": {"name": "task", "schema": SCHEMA}}}
                async with sem:
                    try:
                        data = await chat(client, url, body)
                        t = json.loads(data["choices"][0]["message"]["content"])
                        clean, problems = validate_template(spec, t)
                        stats["tokens"] += data.get("usage", {}).get("completion_tokens", 0)
                    except Exception as e:  # noqa: BLE001 - a bad template is skipped, never fatal
                        clean, problems = None, [f"{type(e).__name__}: {e}"[:300]]
                if clean:
                    clean["problems"] = problems
                    fo.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    fo.flush()
                    stats["ok"] += 1
                    stats["pairs"] += len(clean["pairs"])
                else:
                    fr.write(json.dumps({"tid": spec["tid"], "family": spec["family"], "problems": problems}) + "\n")
                    fr.flush()
                    stats["bad"] += 1
                n = stats["ok"] + stats["bad"]
                if n % 10 == 0 or n == len(specs):
                    dt = time.perf_counter() - t0
                    print(f"[{n}/{len(specs)}] ok={stats['ok']} bad={stats['bad']} pairs={stats['pairs']} "
                          f"{stats['tokens'] / dt:.0f} tok/s  {n / dt * 3600:.0f} templates/h", flush=True)

            await asyncio.gather(*(one(s) for s in specs))


def messages_for(t: dict, state: str, rng: random.Random | None = None, examples: bool = True) -> list[dict]:
    """The teacher's prompt (DecideBench chat format): one worked example per option as earlier turns."""
    shots = list(t["examples"]) if examples else []
    if rng:
        rng.shuffle(shots)
    return chat_messages(state, t["question"], t["options"], [(e["state"], e["gold"]) for e in shots])


async def label(args) -> None:
    out = Path(args.out)
    templates = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = done_tids(out)
    templates = [t for t in templates if t["tid"] not in seen]
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0
    t0 = time.perf_counter()
    print(f"label: {len(templates)} templates ({len(seen)} already done)", flush=True)

    async def score(client: httpx.AsyncClient, t: dict, state: str) -> dict[str, float]:
        # many-option tasks: the chat format repeats every option per example, so label those zero-shot
        body = {"messages": messages_for(t, state, examples=len(t["options"]) <= 8), "max_tokens": 1,
                "temperature": 0, "logprobs": True,
                "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
        data = await chat(client, url, body)
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        seen_lp: dict[str, float] = {}
        for x in top:  # "A" and " A" are different tokens; both count for option A
            seen_lp[x["token"].strip()] = seen_lp.get(x["token"].strip(), 0.0) + math.exp(x["logprob"])
        floor = min(seen_lp.values(), default=1e-6) * 0.5
        raw = {o["key"]: seen_lp.get(LETTERS[i], floor) for i, o in enumerate(t["options"])}
        z = sum(raw.values())
        return {k: v / z for k, v in raw.items()}

    async with httpx.AsyncClient(timeout=600) as client:
        with out.open("a", encoding="utf-8") as fo:
            async def one(t: dict) -> None:
                nonlocal n_done
                async with sem:  # one template at a time per slot, so the shared prompt prefix stays cached
                    try:
                        for p in t["pairs"]:
                            p["teacher_a"] = await score(client, t, p["a"])
                            p["teacher_b"] = await score(client, t, p["b"])
                    except Exception as e:  # noqa: BLE001
                        print(f"label {t['tid']}: {type(e).__name__}: {e}"[:300], flush=True)
                        return
                fo.write(json.dumps(t, ensure_ascii=False) + "\n")
                fo.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(templates):
                    dt = time.perf_counter() - t0
                    print(f"[label {n_done}/{len(templates)}] {n_done / dt * 3600:.0f} templates/h", flush=True)

            await asyncio.gather(*(one(t) for t in templates))


def trigrams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def build(args) -> None:
    from decidebench.dataset import load_items
    from decidebench.fewshot import load_examples

    bench = load_items() + load_examples()
    bench_tri = [trigrams(it.state) for it in bench]
    bench_q = {it.question.strip().lower() for it in bench}

    def near_bench(text: str) -> bool:
        t = trigrams(text)
        return bool(t) and any(len(t & b) / len(t | b) >= 0.5 for b in bench_tri if b)

    templates = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    train, val = [], []
    stats = {"templates": 0, "items_in": 0, "items_kept": 0, "low_confidence": 0, "teacher_overrode_gold": 0,
             "contrastive_pairs_kept": 0, "contaminated": 0}
    for t in templates:
        if t["question"].strip().lower() in bench_q or any(near_bench(e["state"]) for e in t["examples"]):
            stats["contaminated"] += 2 * len(t["pairs"])
            continue
        is_val = int(hashlib.sha1(t["tid"].encode()).hexdigest(), 16) % 100 < args.val_pct
        rec_base = {k: t[k] for k in ("tid", "family", "kind", "domain", "question", "options", "examples")}
        kept_any = False
        for i, p in enumerate(t["pairs"]):
            labels = {}
            for half in ("a", "b"):
                stats["items_in"] += 1
                probs = p.get(f"teacher_{half}")
                if not probs:
                    continue
                label = max(probs, key=probs.get)
                if probs[label] < args.min_conf:
                    stats["low_confidence"] += 1
                    continue
                if near_bench(p[half]):
                    stats["contaminated"] += 1
                    continue
                agree = label == p[f"gold_{half}"]
                stats["teacher_overrode_gold"] += not agree
                labels[half] = label
                # The teacher's answer is the label (distillation). The generator's reason only explains its own
                # intended answer, so it is kept only where the two agree.
                rec = rec_base | {"id": f"{t['tid']}-{i}{half}", "state": p[half], "gold": label,
                                  "why": p[f"why_{half}"] if agree else "", "teacher": probs, "edit": p["edit"],
                                  "designed_gold": p[f"gold_{half}"]}
                (val if is_val else train).append(rec)
                stats["items_kept"] += 1
                kept_any = True
            stats["contrastive_pairs_kept"] += len(labels) == 2 and labels["a"] != labels["b"]
        stats["templates"] += kept_any
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    stats |= {"train_items": len(train), "val_items": len(val)}
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", default=str(HERE / "generated" / "raw.jsonl"))
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--focus", default="", help="comma-separated families to draw more often (weak on our val set)")
    g.add_argument("--focus-weight", type=float, default=3.0)
    lb = sub.add_parser("label")
    lb.add_argument("--inp", default=str(HERE / "generated" / "raw.jsonl"))
    lb.add_argument("--out", default=str(HERE / "generated" / "labeled.jsonl"))
    for p in (g, lb):
        p.add_argument("--url", default="http://127.0.0.1:8080/v1")
        p.add_argument("--concurrency", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("--inp", default=str(HERE / "generated" / "labeled.jsonl"))
    b.add_argument("--out", default=str(HERE / "generated"))
    b.add_argument("--val-pct", type=int, default=5)
    b.add_argument("--min-conf", type=float, default=0.6, help="drop items where the teacher's top answer is below this")
    args = ap.parse_args()
    if args.cmd == "generate":
        asyncio.run(generate(args))
    elif args.cmd == "label":
        asyncio.run(label(args))
    else:
        build(args)


if __name__ == "__main__":
    main()
