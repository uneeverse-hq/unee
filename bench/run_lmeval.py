"""Widely used LLM benchmarks (EleutherAI lm-evaluation-harness) for Unee, with the settings written into the result.

    PYTHONUTF8=1 .venv/Scripts/python bench/run_lmeval.py [--only unee-0.3-0.8b] [--limit 8]

Four passes, each run for every model and skipped when its result already exists:
- core: ARC-Challenge, HellaSwag and TruthfulQA (mc1, mc2) at the harness defaults (0-shot)
- gsm8k: GSM8K at the harness default (5-shot)
- ifeval: IFEval through the model's chat template, thinking off
- mmlu: MMLU, 5-shot. Last, because on an 8 GB card the 2B model spills into shared memory on the longest prompts
  and takes hours; leave it for when nothing else needs the GPU.

To compare another model under the same settings, add its Hugging Face id or folder to MODELS.

Models load in bf16 with Hugging Face transformers; the Unee ones come from the folders staged for release. The
harness output goes to models/lmeval/<model>/<pass>/ (git-ignored) and the scores to bench/results/lmeval.json.
With --limit, both go under models/lmeval-limit<N>/ instead, so a quick check never looks like a result.

The harness has its own venv that borrows torch and transformers from the main one, so nothing in .venv changes:
    py -3.11 -m venv bench/lmeval/.venv
    (write the full path of .venv\\Lib\\site-packages into bench/lmeval/.venv/Lib/site-packages/_main_venv.pth)
    bench/lmeval/.venv/Scripts/python -m pip install "lm-eval[ifeval]"
This script runs under the main venv with the harness venv on PYTHONPATH, because triton looks for its bundled C
compiler next to the running interpreter and Qwen3.5 needs it. Once the datasets are cached, set HF_HUB_OFFLINE=1
and HF_DATASETS_OFFLINE=1: the Hugging Face connection from this laptop drops often.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "bench/lmeval/.venv/Lib/site-packages"
MODELS = {"unee-0.3-0.8b": "models/hf/unee-0.8b", "unee-0.3-2b": "models/hf/unee-2b"}
# MMLU runs one item at a time: the harness keeps logits and a log-softmax for every position, and one of the longest
# 5-shot prompts (about 2,800 tokens x a 248k vocabulary) needs about 2.8 GB for those two tensors alone.
PASSES = {"core": (["arc_challenge", "hellaswag", "truthfulqa_mc1", "truthfulqa_mc2"], ["--batch_size", "8"]),
          "gsm8k": (["gsm8k"], ["--batch_size", "8"]),
          "ifeval": (["ifeval"], ["--batch_size", "8", "--apply_chat_template"]),
          "mmlu": (["mmlu"], ["--batch_size", "1", "--num_fewshot", "5"])}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def result_file(out: Path) -> Path | None:
    found = sorted(out.rglob("results_*.json"))
    return found[-1] if found else None


def scores(out: Path) -> tuple[str, dict]:
    """The harness version and, per task, every numeric metric plus the number of items scored."""
    d = json.loads(result_file(out).read_text(encoding="utf-8"))
    n = d.get("n-samples", {})
    keep = {}
    for task, m in d["results"].items():
        if task.startswith("mmlu_"):  # 57 subjects and 4 category groups; the "mmlu" row carries the score
            continue
        keep[task] = {k.replace(",none", ""): round(v, 4) for k, v in m.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)}
        keep[task]["n"] = n[task]["effective"] if task in n else sum(
            v["effective"] for k, v in n.items() if k.startswith(task + "_"))
    return str(d.get("lm_eval_version")), keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(MODELS))
    ap.add_argument("--limit", type=int, help="a quick check on this many items per task")
    args = ap.parse_args()
    out_root = ROOT / "models" / (f"lmeval-limit{args.limit}" if args.limit else "lmeval")
    summary_path = out_root / "summary.json" if args.limit else ROOT / "bench/results/lmeval.json"
    # The allocator must hand cached blocks back: the longest MMLU prompts push an 8 GB card into Windows' shared
    # memory, and without this the cache stays there and the rest of the pass runs ten times slower.
    env = {**os.environ, "PYTHONPATH": str(HARNESS), "PYTHONWARNINGS": "ignore",
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6"}
    failed = []
    for pname, (tasks, extra) in PASSES.items():
        for name in args.only or MODELS:
            out = out_root / name / pname
            if result_file(out) is None:
                out.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args",
                       f"pretrained={MODELS[name]},dtype=bfloat16,enable_thinking=False", "--tasks", ",".join(tasks),
                       "--device", "cuda:0", "--output_path", str(out), *extra,
                       *(["--limit", str(args.limit)] if args.limit else [])]
                log(f"{name} {pname}: start")
                t0 = time.time()
                with open(out / "run.log", "a", encoding="utf-8") as f:
                    rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
                log(f"{name} {pname}: exit {rc} after {(time.time() - t0) / 60:.1f} min")
            if result_file(out) is None:
                failed.append(f"{name} {pname}")
                log(f"{name} {pname}: FAILED, see {out / 'run.log'}")
                continue
            version, got = scores(out)
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            summary["_settings"] = {"harness": f"lm-eval {version}", "dtype": "bfloat16", "limit": args.limit,
                                    "passes": {p: {"tasks": t, "args": a} for p, (t, a) in PASSES.items()}}
            summary.setdefault(name, {}).update(got)
            summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
            log(f"{name} {pname}: {json.dumps(got)}")
    log("lmeval done" + (f", FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
