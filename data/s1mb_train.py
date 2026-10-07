"""Training (and selection) data from the public bekko-system-one-dataset-v0 train/validation splits, converted to
Unee's training records through exactly the requests S1MB sends at test time.

    bench/S1MB/.venv/Scripts/python data/s1mb_train.py --out data/generated/s1mb_train.jsonl \
        --val-out data/generated/s1mb_val.jsonl

Run from PowerShell with HF_HUB_OFFLINE unset (it downloads). It uses S1MB's own row conversion (case_from_row) and
request builder (questions_for_api, as its TypeSafe adapter calls it), then Unee's own request parser, so a
training prompt is the prompt the server builds for that S1MB-style request.

Rules:
- **Only the train and validation splits.** Test splits (which are S1MB's test set) are never read.
- **Any case whose input hash matches an S1MB test case is dropped as well.**
- **Only subsets whose upstream licence is permissive** (Apache, MIT, BSD, CC BY / BY-SA, CC0, public domain).
  Non-commercial, research-only, access-restricted or unknown-licence sources are skipped, and the list of
  sources used is written next to the output.
- **Questions with more than 8 options** become one tournament round as the server plays it: the target option
  plus up to 7 others.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/S1MB/evaluator/src"))

import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from s1mb.adapters.base import questions_for_api  # noqa: E402
from s1mb.hf_data import case_from_row  # noqa: E402

from unee.systemone import parse, state_value  # noqa: E402

REPO = "hotchpotch/bekko-system-one-dataset-v0"
S1MB_DATA = ROOT / "bench/S1MB/evaluator/data/datasets"
GROUP = 8  # unee/server.py plays more options than this as a tournament of GROUP-option rounds
NOT_FREE = re.compile(r"non-?commercial|research|unknown|unconfirmed|access terms|by-nc|\bnc\b", re.I)
FREE = re.compile(r"apache|\bmit\b|bsd|cc[ -]?by|cc0|public domain|odc-by|cdla-permissive", re.I)


def fetch(filename: str) -> str:
    for attempt in range(10):
        try:
            return hf_hub_download(REPO, filename, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 (the Hub connection here drops often)
            print(f"  retry {attempt} {filename}: {type(e).__name__}", flush=True)
            time.sleep(5)
    raise SystemExit(f"could not download {filename}")


def licences() -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """Licence text per config (README source table) and data files per config and split (README YAML)."""
    text = Path(fetch("README.md")).read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---", 2)[1])
    files = {c["config_name"]: {f["split"]: f["path"] if isinstance(f["path"], list) else [f["path"]]
                                for f in c["data_files"]} for c in front["configs"]}
    lic = {}
    for line in text.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            lic[cells[0].strip("`")] = re.sub(r"\]\([^)]*\)", "]", cells[-1]).replace("[", "").replace("]", "")
    return lic, files


def s1mb_test_hashes() -> set[str]:
    from datasets import load_from_disk
    hashes = set()
    for d in S1MB_DATA.iterdir():
        if d.is_dir() and not d.name.startswith("_") and (d / "test").exists():
            hashes.update(load_from_disk(str(d))["test"]["input_hash"])
    return hashes


def records(row: dict, config: str, rng: random.Random) -> list[dict]:
    case = case_from_row(row)
    request = {"state": case.state, "questions": questions_for_api(case.questions, sort_score=True, structured=True)}
    state, tasks = parse(request)
    out = []
    for q, task in zip(case.questions, tasks):
        probs = case.targets[q.id].probabilities
        if task.kind == "score":  # the API lists score levels lowest value first; keys are their positions
            ordered = sorted(q.options, key=lambda o: o.value)
            teacher = {str(i): probs[o.id] for i, o in enumerate(ordered)}
        else:
            teacher = {o["key"]: probs[o["key"]] for o in task.options}
        options = task.options
        gold = max(teacher, key=teacher.get)
        if len(options) > GROUP:  # one tournament round that contains the target
            others = [o for o in options if o["key"] != gold]
            options = [o for o in options if o["key"] == gold] + rng.sample(others, GROUP - 1)
            rng.shuffle(options)
            kept = {o["key"] for o in options}
            teacher = {k: p for k, p in teacher.items() if k in kept}
            total = sum(teacher.values()) or 1.0
            teacher = {k: p / total for k, p in teacher.items()}
        soft = sorted(teacher.values())[-1] < 0.999
        out.append({"tid": f"s1mb-{config}-{case.case_id}-{q.id}", "family": f"s1mb:{config}", "kind": task.kind,
                    "question": task.question, "options": options, "examples": [],
                    "state": state_value(state), "gold": gold, "teacher": teacher if soft else {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--per-tested", type=int, default=300, help="cases per subset that S1MB also tests")
    ap.add_argument("--per-other", type=int, default=100, help="cases per training-only subset")
    ap.add_argument("--per-val", type=int, default=30, help="validation cases per tested subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    lic, files = licences()
    tested = {d.name for d in S1MB_DATA.iterdir() if d.is_dir() and not d.name.startswith("_")}
    banned = s1mb_test_hashes()
    print(f"{len(files)} configs, {len(tested)} tested by S1MB, {len(banned)} S1MB test hashes", flush=True)
    used, skipped = [], []
    with open(args.out, "w", encoding="utf-8") as fo, open(args.val_out, "w", encoding="utf-8") as fv:
        for config in sorted(files):
            text = lic.get(config, "Unknown")
            if NOT_FREE.search(text) or not FREE.search(text):
                skipped.append((config, text))
                continue
            is_tested = config in tested
            for split, cap, sink in (("train", args.per_tested if is_tested else args.per_other, fo),
                                     ("validation", args.per_val if is_tested else 0, fv)):
                if cap == 0 or split not in files[config]:
                    continue
                rows = []
                for f in files[config][split]:  # convert only a sample (with room for skips) to Python rows
                    table = pq.read_table(fetch(f))
                    rows += table.take(rng.sample(range(table.num_rows), min(table.num_rows, cap * 3))).to_pylist()
                rng.shuffle(rows)
                n_cases = n_items = 0
                for row in rows:
                    if n_cases >= cap:
                        break
                    if row.get("input_hash") in banned:
                        continue
                    try:
                        items = records(row, config, rng)
                    except Exception:  # noqa: BLE001 (ranking decisions and malformed rows are not trainable)
                        continue
                    for r in items:
                        sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_cases += 1
                    n_items += len(items)
                print(f"{config:55s} {split:10s} {n_cases:4d} cases {n_items:5d} items", flush=True)
            used.append((config, text))
    Path(args.out).with_suffix(".sources.json").write_text(
        json.dumps({"used": used, "skipped": skipped}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"used {len(used)} configs, skipped {len(skipped)} for licence", flush=True)


if __name__ == "__main__":
    main()
