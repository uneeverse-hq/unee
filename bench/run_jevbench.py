"""JevBench (github.com/model-collapse/jev-bench, Apache-2.0) with Unee as a backend in JevBench's own harness:
its data loading, evaluation loop and metrics are used unchanged, so numbers compare with its leaderboard.

    git clone --depth 1 https://github.com/model-collapse/jev-bench.git bench/jev-bench
    PYTHONUTF8=1 .venv/Scripts/python bench/run_jevbench.py --base-url http://127.0.0.1:8090/v1 --tier gold \
        --out bench/results/jevbench-<name>.json

Unee runs zero-shot here (JevBench has no worked examples). Questions with more than MAX_GROUP options are decided
as a tournament: the best option of each group of MAX_GROUP goes to the next round.
Anti-overfitting rule: this benchmark is for final models only; no training or selection decision uses it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "jev-bench"))
sys.path.insert(0, str(HERE.parent))

import httpx  # noqa: E402

import bench_eval  # noqa: E402  (JevBench's harness)
from unee.prompt import LETTERS, build_messages  # noqa: E402
from unee.systemone import NOUL_DEFAULT  # noqa: E402

MAX_GROUP = 6  # Unee was trained on 2-6 options


class Unee:
    def __init__(self, model=None, base_url=None, **_):
        self.url = (base_url or "http://127.0.0.1:8090/v1").rstrip("/") + "/chat/completions"
        self.client = httpx.Client(timeout=600)

    def probs(self, state, question: str, options: list[dict]) -> dict[str, float]:
        body = {"messages": build_messages(state, question, options), "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
        r = self.client.post(self.url, json=body)
        r.raise_for_status()
        seen: dict[str, float] = {}
        for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            seen[t["token"].strip()] = seen.get(t["token"].strip(), 0.0) + math.exp(t["logprob"])
        floor = min(seen.values(), default=1e-6) * 0.5
        raw = {o["key"]: seen.get(LETTERS[i], floor) for i, o in enumerate(options)}
        z = sum(raw.values())
        return {k: v / z for k, v in raw.items()}

    def decide(self, state, question: str, options: list[dict]) -> str:
        while len(options) > MAX_GROUP:  # tournament rounds
            winners = []
            for i in range(0, len(options), MAX_GROUP):
                group = options[i:i + MAX_GROUP]
                if len(group) == 1:
                    winners += group
                    continue
                p = self.probs(state, question, group)
                best = max(p, key=p.get)
                winners.append(next(o for o in group if o["key"] == best))
            options = winners
        p = self.probs(state, question, options)
        return max(p, key=p.get)

    def predict(self, row):
        q, t = row["question"], row["type"]
        question = q.get("instructions", "")
        if t == "noul":
            options = [{"key": "true", "description": NOUL_DEFAULT["true"]},
                       {"key": "false", "description": NOUL_DEFAULT["false"]}]
            return "yes" if self.decide(row["state"], question, options) == "true" else "no"
        options = [{"key": k, "description": v} for k, v in bench_eval.candidates(row)]
        return self.decide(row["state"], question, options)


if __name__ == "__main__":
    bench_eval.BACKENDS["unee"] = Unee
    if "--backend" not in sys.argv:
        sys.argv += ["--backend", "unee"]
    if "--data" not in sys.argv:
        sys.argv += ["--data", str(HERE / "jev-bench" / "data")]
    bench_eval.main()
