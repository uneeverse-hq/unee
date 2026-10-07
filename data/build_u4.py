"""Assemble the Unee 0.4 training file: S1MB-style rows that 0.3 never saw, plus replay.

    PYTHONUTF8=1 .venv/Scripts/python data/build_u4.py

Takes s1mb_train_u4.jsonl (a new, larger sample from data/s1mb_train.py with another seed), drops every row
whose id is already in train_u3.jsonl, and caps each subset at CAP items. It adds a 30% replay of 0.2's own decision
data. Chat stays chat_u3.jsonl.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

G = Path(__file__).resolve().parent / "generated"
CAP = 1500


def rows(name: str) -> list[dict]:
    p = G / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def main() -> None:
    rng = random.Random(4)
    seen = {r["tid"] for r in rows("train_u3.jsonl") if "tid" in r}
    fresh = [r for r in rows("s1mb_train_u4.jsonl") if r["tid"] not in seen]
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in fresh:
        by_family[r["family"]].append(r)
    s1mb = [r for fam in by_family.values() for r in (rng.sample(fam, CAP) if len(fam) > CAP else fam)]
    own = rows("train_all.jsonl")
    train = s1mb + rng.sample(own, round(len(own) * 0.3))
    rng.shuffle(train)
    (G / "train_u4.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    print(f"train_u4: {len(train)} ({len(s1mb)} new S1MB-style rows, {len(fresh)} unseen before the cap)")


if __name__ == "__main__":
    main()
