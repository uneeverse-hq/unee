"""Assemble the Unee 0.3 training files (continuing from the 0.2 models).

    PYTHONUTF8=1 .venv/Scripts/python data/build_v3.py

- **Decisions** (`train_u3.jsonl`; "u3" because older experiments already used "v3" names): s1mb_train.jsonl (public train splits, S1MB-style requests) with at most
  1,500 items per subset, plus
  a 30% replay of 0.2's own decision data (train_all.jsonl), so 0.2's skills are kept. Shuffled.
- **Chat** (`chat_u3.jsonl`):
  - the teacher's grounded answers and refusals (chat_teacher.jsonl kinds "knowledge" and "summary")
  - SQuAD 2.0 grounded answers (qa_grounded.jsonl)
  - general chat answered by the base model itself (chat_distill.jsonl)

  0.2's teacher-written general chat (asked to be concise) is left out: a blind judge preferred the base model's
  answers to it.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

G = Path(__file__).resolve().parent / "generated"
CAP = 1500  # items per S1MB-style subset


def rows(name: str) -> list[dict]:
    p = G / name
    # split on "\n" only: str.splitlines() also breaks on U+2028 and similar characters inside JSON strings
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def main() -> None:
    rng = random.Random(3)
    s1mb, own = rows("s1mb_train.jsonl"), rows("train_all.jsonl")
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in s1mb:
        by_family[r["family"]].append(r)
    # Some subsets ask many questions per case (go_emotions: 28), so a case cap alone lets five subsets fill 40%
    # of the data; cap each subset's items so every task counts.
    s1mb = [r for fam in by_family.values() for r in (rng.sample(fam, CAP) if len(fam) > CAP else fam)]
    replay = rng.sample(own, round(len(own) * 0.3))
    train = s1mb + replay
    rng.shuffle(train)
    (G / "train_u3.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    teacher = [r for r in rows("chat_teacher.jsonl") if r["kind"] in ("knowledge", "summary")]
    chat = teacher + rows("qa_grounded.jsonl") + rows("chat_distill.jsonl")
    rng.shuffle(chat)
    (G / "chat_u3.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat), encoding="utf-8")
    print(f"train_u3: {len(train)} ({len(s1mb)} S1MB-style, {len(replay)} replay); chat_u3: {len(chat)}",
          dict(collections.Counter(r.get("kind", "general") for r in chat)))


if __name__ == "__main__":
    main()
