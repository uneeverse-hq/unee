"""S1MB "Task Avg" for a local run, computed the way S1MB's SCORING.md defines it, next to published leaderboard rows.

    PYTHONUTF8=1 .venv/Scripts/python bench/s1mb_summary.py bench/S1MB/evaluator/data/results/unee-r3-0.8b

Task Avg is computed in three steps:
1. Each benchmark's baseline-adjusted score (clipped to 0..1) as S1MB saves it; benchmarks where the adjustment is
   undefined are left out, as S1MB does.
2. The mean over benchmarks within each task (choice, noul, score).
3. The mean of the three task means, times 100.

A run counts only when every benchmark in it is complete; anything else is reported as partial and never compared.
Borda needs every model's per-benchmark scores, so it is not computed here.

Several folders joined with "+" are one model's results: a later folder replaces a benchmark that is incomplete in an
earlier one (used when long-document benchmarks are rerun with a longer context, same model and prompts):
    bench/s1mb_summary.py results/unee-r3-0.8b+results/unee-r3-0.8b-long
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# S1MB leaderboard, full English suite, snapshot of 2026-09-30 (hotchpotch.dev S1MB article): Task Avg (0-100).
PUBLISHED = {"Jev 1.13": 59.59, "Open-Jev-9B": 52.10, "bekko-system-one-v0-400m": 50.60, "Kev-0.8b": 18.68,
             "von": 16.21, "Laya": 13.36, "JevForge-0.8B": 1.92}


FLAGGED = Path(__file__).resolve().parent / "results/contamination-s1mb.json"  # tools/contamination_check.py


def flagged_datasets() -> set[str]:
    """S1MB datasets with any test input overlapping Unee's training text (exact or 13-gram), for the clean score."""
    return {f"datasets/{r['dataset']}" for r in json.loads(FLAGGED.read_text())} if FLAGGED.exists() else set()


def summarise(*run_dirs: Path, exclude: set[str] | None = None) -> dict:
    """Task Avg as S1MB defines it. With `exclude` (dataset paths), those benchmarks are left out: the "clean"
    score, which also stays comparable between models only on the same remaining benchmarks."""
    results: dict[str, dict] = {}
    for d in run_dirs:  # a later folder fills in benchmarks an earlier one left incomplete
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if exclude and r["benchmark"].get("dataset") in exclude:
                continue
            prev = results.get(r["benchmark"]["id"])
            if prev is None or prev.get("status") != "complete":
                results[r["benchmark"]["id"]] = r
    by_task: dict[str, list[float]] = {"choice": [], "noul": [], "score": []}
    incomplete = []
    for bid, r in sorted(results.items()):
        if r.get("status") != "complete":
            incomplete.append(bid)
        score = r.get("metrics", {}).get("baseline_adjusted_score")
        if score is not None:
            by_task[r["benchmark"]["task"]].append(score)
    tasks = {t: round(statistics.mean(v) * 100, 2) if v else None for t, v in by_task.items()}
    complete = not incomplete and all(v is not None for v in tasks.values())
    return {"run": "+".join(Path(d).name for d in run_dirs), "benchmarks": sum(map(len, by_task.values())), "complete": complete,
            "incomplete": incomplete[:10], "tasks": tasks,
            "task_avg": round(statistics.mean(tasks.values()), 2) if complete else None}


def main() -> None:
    clean = "--clean" in sys.argv
    for d in [a for a in sys.argv[1:] if a != "--clean"]:
        s = summarise(*[Path(x) for x in d.split("+")], exclude=flagged_datasets() if clean else None)
        print(json.dumps(s))
        if clean:
            continue  # published rows are full-suite scores; a clean score compares only with other clean scores
        if s["task_avg"] is not None:
            rows = sorted([*PUBLISHED.items(), (s["run"], s["task_avg"])], key=lambda x: -x[1])
            for name, v in rows:
                print(f"  {v:6.2f}  {name}")


if __name__ == "__main__":
    main()
