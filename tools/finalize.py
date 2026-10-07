"""Pick the release models on our own val set, then score only the picks on DecideBench.

    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python tools/finalize.py --tag r2

Candidates are merged-model directories under models/ (single runs and soups). Selection uses val accuracy only;
DecideBench is run once per size for the chosen model. Writes models/finalize-<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SIZES = {
    "0.8b": ["soup3", "soup4r", "soup3r", "v1", "rb-08b"],
    "2b": ["e2", "e2b", "e2c", "soup-2b", "soup2b3"],
}


def run(*args: str) -> str:
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if not (ROOT / "models/soup2b3/merged").exists() and (ROOT / "models/e2c/merged").exists():
        run("tools/soup.py", "--out", "models/soup2b3/merged", "models/e2/merged", "models/e2b/merged", "models/e2c/merged")
    out: dict = {"val": {}, "chosen": {}, "decidebench": {}}
    for size, names in SIZES.items():
        for n in names:
            if not (ROOT / f"models/{n}/merged").exists():
                continue
            res = run("train/eval_val.py", "--model", f"models/{n}/merged", "--val", "data/generated/val.jsonl")
            try:
                out["val"][n] = json.loads(res[res.index("{"):])["t1"]["acc"]
            except ValueError:
                out["val"][n] = None
            print(size, n, out["val"][n], flush=True)
        scored = {n: out["val"][n] for n in names if out["val"].get(n) is not None}
        best = max(scored, key=scored.get)
        out["chosen"][size] = best
        run("bench/run_local.py", "--model", f"models/{best}/merged", "--device", "cuda", "--format", "compact",
            "--name", f"unee-final-{args.tag}-{size}")
        s = json.loads((ROOT / f"bench/results/unee-final-{args.tag}-{size}.summary.json").read_text())
        out["decidebench"][size] = {k: s[k] for k in ("acc", "pair_acc", "ece", "by_family")}
        print("CHOSEN", size, best, out["decidebench"][size], flush=True)
    (ROOT / f"models/finalize-{args.tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
