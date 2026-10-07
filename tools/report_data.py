"""Build every number on the website's Unee technical page from raw result files, and write the site's data module.

    PYTHONUTF8=1 .venv/Scripts/python tools/report_data.py --version 0.2 \
        --web ../uneeverse-web/src/content/unee-report.ts

Sources:
- **DecideBench leaderboard:** the tables in bench/decidebench/README.md (overall plus per family).
- **Unee on DecideBench:** bench/results/<run>.summary.json and <run>.jsonl (bf16, the benchmark's protocol).
- **Cost per 1M decisions:** DecideBench's own method for self-hosted entries, GPU-hours × hourly price ÷ items at 4
  requests in flight. Throughput comes from bench/results/<gguf run>.speed.json (llama.cpp on this laptop's RTX
  4070), priced at the L4 rate DecideBench uses ($0.81/h). It is a stand-in, not a measurement on an L4; the page
  says so.
- **Independent checks:** JevBench (bench/results/jevbench-unee-*-gold.json). When present, also S1MB, MASSIVE and
  the knowledge eval, taken from models/release_r3.json and bench/results.
- **Leakage and the clean S1MB score:** bench/results/contamination.json (tools/contamination_check.py), and the S1MB
  Task Avg without every benchmark it flags (bench/s1mb_summary.py --clean).
- **Strict mode:** bench/results/knowledge-<tag>-<size>-draft.json (no check) and -strict.json (threshold 0.5), the
  same drafts judged with and without the check.
- **General LLM benchmarks:** bench/results/lmeval.json (bench/run_lmeval.py, lm-evaluation-harness), the headline
  metric of each task only.
- **CPU and browser speed:** bench/results/potato-*.speed.json and the measured browser range.

Missing inputs leave their field null, and the page hides the matching section, so nothing unmeasured is shown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench/results"
sys.path.insert(0, str(ROOT / "bench/decidebench"))
sys.path.insert(0, str(ROOT / "bench"))
from s1mb_summary import PUBLISHED as S1MB_PUBLISHED, flagged_datasets, summarise  # noqa: E402

S1MB_RESULTS = ROOT / "bench/S1MB/evaluator/data/results"

L4_HOURLY = 0.81  # DecideBench's price for its NVIDIA L4 entries (results/v1/meta/*.json)
SIZES = {"Unee 0.8B": 0.75, "Unee 2B": 1.9}  # billions of parameters


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def s1mb_clean(run: str) -> dict | None:
    """S1MB Task Avg without the benchmarks whose test inputs overlap Unee's training text."""
    dirs = [S1MB_RESULTS / part for part in run.split("+")]
    if not all(d.exists() for d in dirs):
        return None
    s = summarise(*dirs, exclude=flagged_datasets())
    return {"benchmarks": s["benchmarks"], "task_avg": s["task_avg"]} if s["complete"] else None


LMEVAL_METRICS = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "truthfulqa_mc1": "acc",
                  "truthfulqa_mc2": "acc", "gsm8k": "exact_match,strict-match",
                  "ifeval": ("prompt_level_strict_acc", "inst_level_strict_acc")}


def lmeval(version: str) -> dict | None:
    """Unee's lm-evaluation-harness scores (0..1), one headline metric per task, and the harness version."""
    raw = read(RES / "lmeval.json")
    if not raw:
        return None
    out = {}
    for size in ("0.8b", "2b"):
        run = raw.get(f"unee-{version}-{size}") or {}
        scores = {}
        for task, metric in LMEVAL_METRICS.items():
            for m in metric if isinstance(metric, tuple) else (metric,):
                if m in run.get(task, {}):
                    scores[f"{task}:{m}" if isinstance(metric, tuple) else task] = run[task][m]
        if scores:
            out[size] = scores
    return {"harness": raw["_settings"]["harness"], "unee": out} if out else None


def strict_pair(key: str) -> dict | None:
    """The same knowledge drafts judged without and with strict mode, summary fields only."""
    pair = {v: read(RES / f"knowledge-{key}-{f}.json") for v, f in (("draft", "draft"), ("strict", "strict"))}
    if not all(pair.values()):
        return None
    return {v: {k: x for k, x in d.items() if k != "per_question"} for v, d in pair.items()}


def pct(s: str) -> float | None:
    m = re.match(r"([\d.]+)%", s.strip())
    return float(m.group(1)) if m else None


def decidebench_tables() -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Rows of the README's main table and its per-family table."""
    lines = (ROOT / "bench/decidebench/README.md").read_text(encoding="utf-8").splitlines()
    main, fam, fam_cols = [], {}, []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or cells[0] in ("Entry", "Random") or set(cells[0]) <= {"-"}:
            if line.startswith("| Entry") and "`action_review`" in line:
                fam_cols = [c.strip("`") for c in cells[1:]]
            continue
        if len(cells) == 7 and cells[1].endswith("%"):
            cost = re.match(r"\$([\d,.]+)", cells[4])
            lat = re.match(r"([\d,]+) ms", cells[5])
            main.append({"name": cells[0], "accuracy": pct(cells[1]), "pair": pct(cells[2]),
                         "cost_per_m": float(cost.group(1).replace(",", "")) if cost else None,
                         "latency_ms": float(lat.group(1).replace(",", "")) if lat else None,
                         "hosted": "self-hosted" not in cells[0]})
        elif fam_cols and len(cells) == len(fam_cols) + 1:
            fam[cells[0]] = {c: pct(v) for c, v in zip(fam_cols, cells[1:])}
    return main, fam


def params_from_name(name: str) -> float | None:
    """Total parameters stated in an entry's name; "E2B"-style names state effective size only, so none."""
    m = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*([BbMm])\b", name)
    if not m or re.search(r"\bE\d+B\b", name):
        return None
    return float(m.group(1)) / (1000 if m.group(2) in "Mm" else 1)


def reliability(run: str, bins: int = 10) -> list[dict] | None:
    """Confidence of the top option against how often it is right, in equal-width bins."""
    from decidebench.dataset import load_items
    path = RES / f"{run}.jsonl"
    if not path.exists():
        return None
    gold = {it.id: it.gold for it in load_items()}
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("probs"):
            top = max(r["probs"], key=r["probs"].get)
            pts.append((r["probs"][top], top == gold[r["item_id"]]))
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in pts if lo < c <= hi or (b == 0 and c == 0)]
        if inside:
            out.append({"confidence": round(sum(c for c, _ in inside) / len(inside), 4),
                        "accuracy": round(sum(ok for _, ok in inside) / len(inside), 4), "n": len(inside)})
    return out


def speed_rows(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(RES.glob(pattern)):
        d = read(f)
        rows.append({"name": d.get("name", f.stem), **{k: d[k] for k in ("few_shot_c1", "few_shot_c4", "zero_shot_c1")
                                                       if k in d}})
    return rows


def jevbench(size: str) -> dict | None:
    rows = read(RES / f"jevbench-unee-{size}-gold.json")
    if not rows:
        return None
    by_type = {}
    for t in ("choice", "noul", "score"):
        got = [r for r in rows if r["type"] == t and not r["errored"]]
        if got:
            by_type[t] = round(sum(r["correct"] for r in got) / len(got) * 100, 1)
    return {"n": len(rows), "overall": round(sum(r["correct"] for r in rows) / len(rows) * 100, 1), **by_type}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="the Unee version these numbers describe, e.g. 0.2")
    ap.add_argument("--decidebench", nargs=2, default=["unee-final-r2-0.8b", "unee-final-r2-2b"],
                    help="bf16 DecideBench runs for 0.8B and 2B")
    ap.add_argument("--served", nargs=2, default=["unee-0.8b-q4km-cuda-c4", "unee-2b-q4km-cuda-c4"],
                    help="4-bit llama.cpp DecideBench runs (the system whose throughput is priced)")
    ap.add_argument("--speed", nargs=2, default=["unee-0.8b-q4km-cuda", "unee-2b-q4km-cuda"])
    ap.add_argument("--web", help="write the website's TypeScript data module here")
    ap.add_argument("--tag", default="u3", help="release tag of the newest version (u3 = 0.3, r3 = 0.2)")
    args = ap.parse_args()

    board, fam = decidebench_tables()
    unee = []
    for label, run, served, speed in zip(("Unee 0.8B", "Unee 2B"), args.decidebench, args.served, args.speed):
        s, q, sp = read(RES / f"{run}.summary.json"), read(RES / f"{served}.summary.json"), read(RES / f"{speed}.speed.json")
        tput = sp["few_shot_c4"]["throughput_per_s"] if sp else None
        unee.append({
            "name": label, "params_b": SIZES[label],
            "accuracy": round(s["acc"] * 100, 2) if s else None, "pair": round(s["pair_acc"] * 100, 2) if s else None,
            "ece": round(s["ece"], 3) if s else None,
            "families": {k: round((v["acc"] if isinstance(v, dict) else v) * 100, 1) for k, v in s["by_family"].items()} if s else None,
            "served_accuracy": round(q["acc"] * 100, 2) if q else None,
            "throughput_per_s": tput,
            "cost_per_m": round(L4_HOURLY * 1e6 / 3600 / tput, 2) if tput else None,
            "latency_ms_c1": sp["few_shot_c1"]["p50_ms"] if sp else None,
            "reliability": reliability(run),
        })
    for row in board:
        row["params_b"] = params_from_name(row["name"])
    rel = read(ROOT / f"models/release_{args.tag}.json") or {}
    keys = [f"{args.tag}-0.8b", f"{args.tag}-2b", "r3-0.8b", "r3-2b", "old-0.8b", "old-2b", "base-0.8b"]
    data = {
        "version": args.version,
        "decidebench": {"board": board, "families": fam, "unee": unee, "l4_hourly": L4_HOURLY},
        "jevbench": {"unee": {"0.8B": jevbench("0.8b"), "2B": jevbench("2b")},
                     "board": [{"name": "Jev", "overall": 88.8}, {"name": "gpt-oss-20b", "overall": 88.4},
                               {"name": "bart-mnli", "overall": 54.3}, {"name": "Laya", "overall": 53.9}],
                     "board_note": "JevBench leaderboard, 232-row gold sample (not the same rows as Unee's 1,682)."},
        "potato": speed_rows("potato-*.speed.json"),
        "massive": {k: read(RES / f"massive-{k}.json") for k in keys if (RES / f"massive-{k}.json").exists()} or None,
        "knowledge": {k: read(RES / f"knowledge-{k}.json") for k in keys if (RES / f"knowledge-{k}.json").exists()}
                     or None,
        "knowledge_strict": {k: strict_pair(k) for k in keys[:2] if strict_pair(k)} or None,
        "lmeval": lmeval(args.version),
        "s1mb": {"unee": rel["s1mb"], "published": [{"name": k, "task_avg": v} for k, v in S1MB_PUBLISHED.items()],
                 "published_note": "S1MB leaderboard, full English suite, snapshot of 2026-09-30.",
                 "clean": {size: s1mb_clean(r["run"]) for size, r in rel["s1mb"].items()},
                 "leakage": (read(RES / "contamination.json") or {}).get("S1MB test")}
                if rel.get("s1mb") else None,
        "max_9b": rel.get("max_9b") or None,
    }
    out = ROOT / "docs/report/report-data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if args.web:
        Path(args.web).write_text(
            "/*\n * Generated by tools/report_data.py in the Unee project from raw benchmark files. Regenerate it, do not\n"
            " * edit it: every number on /unee/technical traces back to a result file there.\n */\n\n"
            f"export const report = {json.dumps(data, indent=2, ensure_ascii=False)} as const;\n", encoding="utf-8")
        print(f"wrote {args.web}")


if __name__ == "__main__":
    main()
