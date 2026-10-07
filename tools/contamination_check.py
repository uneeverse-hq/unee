"""Leakage check: does any evaluation item, or a near copy of one, appear in Unee's training data?

    PYTHONUTF8=1 .venv/Scripts/python tools/contamination_check.py

Training text is pooled from every training file: decision states and questions, few-shot example states, chat
prompts, knowledge passages and replies. Each evaluation set is compared with that pool in two ways:
- **exact:** the item's normalised text (lowercase, letters and digits only, single spaces) equals a training text.
- **13-gram:** at least half of the item's 13-word sequences occur in the training pool. This is the usual test for
  near-duplicate contamination; items under 13 words are only checked exactly.

Evaluation sets:
- DecideBench v1.1 (states and worked examples)
- the S1MB test set (all 137 benchmarks' states)
- our selection sets (val, val_ml, s1mb_val)
- the knowledge eval
- JevBench (every tier's states), when bench/jev-bench is checked out

Writes bench/results/contamination.json and prints a table. It also writes data/generated/s1mb_val_clean.jsonl, the
S1MB-style selection set without its overlapping rows, so a model pick can be re-checked on clean rows only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data/generated"
N = 13


def norm(text) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    return " ".join(re.findall(r"\w+", text.lower()))


def grams(t: str) -> set[int]:
    """Hashes of the text's 13-word sequences (hashes keep the training pool's set small enough for memory)."""
    w = t.split()
    return {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()] if path.exists() else []


def train_texts() -> list[str]:
    out = []
    for name in ("train_all.jsonl", "train_u3.jsonl"):
        for r in jsonl(G / name):
            out += [r.get("state"), r.get("question")] + [e.get("state") for e in r.get("examples", [])]
    for name in ("chat_u3.jsonl", "chat_teacher.jsonl", "chat_distill.jsonl", "qa_grounded.jsonl"):
        for r in jsonl(G / name):
            out += [m.get("content") for m in r.get("messages", [])] + [r.get("prompt"), r.get("response")]
    return [t for t in (norm(x) for x in out if x) if t]


def eval_sets() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    db = []
    for f in sorted((ROOT / "bench/decidebench/data/v1").glob("*.jsonl")):
        for r in jsonl(f):
            db.append(r.get("state"))
    for f in sorted((ROOT / "bench/decidebench/data/v1/examples").glob("*.jsonl")):
        for r in jsonl(f):
            db.append(r.get("state"))
    sets["DecideBench v1.1"] = db
    try:
        from datasets import load_from_disk
        s1 = []
        for d in sorted((ROOT / "bench/S1MB/evaluator/data/datasets").iterdir()):
            if d.is_dir() and not d.name.startswith("_") and (d / "test").exists():
                s1 += [r["state_json"] for r in load_from_disk(str(d))["test"]["input"]]
        sets["S1MB test"] = s1
    except Exception as e:  # noqa: BLE001
        print(f"S1MB test not loaded: {e}", file=sys.stderr)
    for name, f in (("val (selection)", "val.jsonl"), ("val_ml (selection)", "val_ml.jsonl"),
                    ("s1mb_val (selection)", "s1mb_val.jsonl")):
        sets[name] = [r.get("state") for r in jsonl(G / f)]
    sets["knowledge eval"] = [r["messages"][-1]["content"] for r in jsonl(G / "kb_eval.jsonl")]
    jb = ROOT / "bench/jev-bench/data"
    if jb.exists():
        sets["JevBench"] = [r.get("state") for f in sorted(jb.rglob("*.jsonl")) for r in jsonl(f)]
    return {k: [t for t in (norm(x) for x in v if x) if t] for k, v in sets.items()}


def main() -> None:
    pool = train_texts()
    exact = set(pool)
    pool_grams: set[int] = set()
    for t in pool:
        pool_grams |= grams(t)
    print(f"training pool: {len(pool)} texts, {len(pool_grams)} distinct {N}-grams")

    def flagged(t: str) -> bool:
        g = grams(t)
        return t in exact or bool(g and len(g & pool_grams) / len(g) >= 0.5)

    report = {}
    for name, items in eval_sets().items():
        ex = sum(t in exact for t in items)
        near = sum(flagged(t) for t in items) - ex
        report[name] = {"items": len(items), "exact": ex, "near_13gram": near,
                        "rate": round((ex + near) / max(len(items), 1), 4)}
        print(f"{name:24s} {len(items):6d} items  exact {ex:4d}  near {near:4d}  ({report[name]['rate']:.2%})")
    (ROOT / "bench/results/contamination.json").write_text(json.dumps(report, indent=1) + "\n")
    rows = jsonl(G / "s1mb_val.jsonl")
    clean = [r for r in rows if not flagged(norm(r.get("state")))]
    (G / "s1mb_val_clean.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + "\n",
                                            encoding="utf-8")
    print(f"s1mb_val_clean.jsonl: {len(clean)} of {len(rows)} rows")


if __name__ == "__main__":
    main()
