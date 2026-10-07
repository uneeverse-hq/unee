# bench/

Local DecideBench runs, used to pick the backbone and track progress.

## Setup

DecideBench is a separate checkout. It's git-ignored here.

```bash
git clone --depth 1 https://github.com/choyiny/decidebench.git bench/decidebench
```

## Run

`PYTHONUTF8=1` is needed on Windows because the data files are UTF-8.

```bash
PYTHONUTF8=1 .venv/Scripts/python bench/run_local.py --model Qwen/Qwen3.5-0.8B --device cuda
PYTHONUTF8=1 .venv/Scripts/python bench/run_local.py --model Qwen/Qwen3.5-0.8B --device cpu --threads 4 --limit 40
```

The runner uses DecideBench's own chat prompt with one worked example per option. It scores each option from the next-token logits, which is one forward pass per item. It writes these to `bench/results/`:
- `<name>.jsonl`: per-item predictions, in DecideBench's format
- `<name>.summary.json`: accuracy, pair accuracy, ECE, latency and per-family accuracy

**Contamination rule:** nothing under `bench/decidebench/data/` may go into training data. That covers the 400 test items and the 297 worked examples.
