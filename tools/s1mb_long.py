"""Rerun the S1MB benchmarks a release run left incomplete, with one 32k-token slot instead of four 4k ones.

    PYTHONUTF8=1 .venv/Scripts/python tools/s1mb_long.py

The release pass serves four parallel slots sharing a 16k context, so each request gets 4,096 tokens, and the
long-document benchmarks (contracts, research papers) fail there. This reruns only those, same model, adapter and
prompts, into bench/S1MB/evaluator/data/results/unee-r3-<size>-long, then records the combined summary in
models/release_r3.json. Results pages should say the long benchmarks came from this second run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from s1mb_summary import summarise  # noqa: E402

STATE = ROOT / "models/release_r3.json"
S1MB_PY = ROOT / "bench/S1MB/.venv/Scripts/python.exe"
RESULTS = ROOT / "bench/S1MB/evaluator/data/results"


def main() -> None:
    s = json.loads(STATE.read_text())
    for size in ("0.8b", "2b"):
        main_dir, long_dir = RESULTS / f"unee-r3-{size}", RESULTS / f"unee-r3-{size}-long"
        todo = summarise(main_dir)["incomplete"]
        if todo and not long_dir.exists():
            url = "http://127.0.0.1:8022"
            srv = subprocess.Popen(
                [sys.executable, "-m", "unee.cli", "serve", "--model", s["gguf"][size], "--llama-server",
                 str(ROOT / "tools/llama.cpp/cuda/llama-server.exe"), "--gpu-layers", "99", "--slots", "1",
                 "--ctx", "32768", "--port", "8022"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(150):
                    try:
                        if httpx.get(url + "/health", timeout=2).json().get("status") == "ok":
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(2)
                bench = [a for b in todo for a in ("--benchmark", b)]
                subprocess.run([str(S1MB_PY), "bench/run_s1mb.py", "--unee-url", url, "--", "--data-dir",
                                "bench/S1MB/evaluator/data", "run", "--offline-dataset", "--adapter", "typesafe",
                                "--model", f"unee-{size}", "--run-id", f"unee-r3-{size}-long", *bench], cwd=ROOT)
            finally:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
                time.sleep(3)
        merged = summarise(main_dir, long_dir) if long_dir.exists() else summarise(main_dir)
        s.setdefault("s1mb", {})[size] = {**s["s1mb"].get(size, {}), **merged, "long_rerun": todo,
                                          "results": [str(d.relative_to(ROOT)) for d in (main_dir, long_dir)
                                                      if d.exists()]}
        STATE.write_text(json.dumps(s, indent=1) + "\n")
        print(size, json.dumps(merged))


if __name__ == "__main__":
    main()
