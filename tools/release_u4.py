"""Unee 0.4 release pass (models/chain_u4.ps1 runs it after training). Same rule and steps as release_u4.py.

    PYTHONUTF8=1 .venv/Scripts/python tools/release_u4.py

Selection rule, fixed on 2026-10-06 before any 0.3 result existed. Each size's release model is the candidate with
the best mean of three held-out accuracies, weighted equally:
- val (English, our tasks)
- val_ml (translated)
- val_s1mb (public validation splits of the S1MB-style data, never S1MB test)

The candidates are the 0.2 model, the u3 run continued from it, and their 50/50 weight average. No benchmark takes
part in choosing.

Steps, in priority order. Each finished step is recorded in models/release_u4.json, so a rerun skips it.
1. Choose.
2. DecideBench.
3. GGUF and ONNX.
4. Full S1MB (long-document benchmarks rerun with one 32k slot).
5. GPU speed plus served 4-bit DecideBench.
6. MASSIVE.
7. CPU recording.
8. chat60 answers, then the teacher.
9. Knowledge answers and their judging.
10. Chat judge against base.
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

PY = sys.executable
S1MB_PY = ROOT / "bench/S1MB/.venv/Scripts/python.exe"
BASH = r"C:\Program Files\Git\bin\bash.exe"
LLAMA_CUDA = str(ROOT / "tools/llama.cpp/cuda/llama-server.exe")
LLAMA_CPU = str(ROOT / "tools/llama.cpp/cpu/llama-server.exe")
STATE = ROOT / "models/release_u4.json"
LOG = ROOT / "models/release_u4.log"
S1MB_RESULTS = ROOT / "bench/S1MB/evaluator/data/results"
SOUPS = {"mix4-08b": ["mix-08b", "u4-08b"], "mix4-2b": ["u3-2b", "u4-2b"]}
CANDIDATES = {"0.8b": ["mix-08b", "u4-08b", "mix4-08b"], "2b": ["u3-2b", "u4-2b", "mix4-2b"]}
VALS = {"val": "data/generated/val.jsonl", "val_ml": "data/generated/val_ml.jsonl",
        "val_s1mb": "data/generated/s1mb_val.jsonl"}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], py: str = PY) -> int:
    log("run " + " ".join(args))
    with open(LOG, "a", encoding="utf-8") as f:
        return subprocess.run([py, *args] if py else args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1) + "\n")


def val_acc(name: str, which: str) -> float | None:
    out = ROOT / f"models/{name}/{which}.json"
    if not out.exists():
        run(["train/eval_val.py", "--model", f"models/{name}/merged", "--val", VALS[which], "--out", str(out)])
    try:
        return json.loads(out.read_text())["t1"]["acc"]
    except (OSError, KeyError, ValueError):
        return None


class Server:
    """`unee serve` on llama.cpp; the whole process tree is killed on exit."""

    def __init__(self, gguf: str, port: int = 8020, cpu: bool = False, slots: int = 4, ctx: int = 16384) -> None:
        self.url = f"http://127.0.0.1:{port}"
        build = [LLAMA_CPU, "--threads", "4"] if cpu else [LLAMA_CUDA, "--gpu-layers", "99"]
        self.proc = subprocess.Popen(
            [PY, "-m", "unee.cli", "serve", "--model", gguf, "--llama-server", *build, "--slots", str(slots),
             "--ctx", str(ctx), "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __enter__(self):
        for _ in range(150):
            try:
                if httpx.get(self.url + "/health", timeout=2).json().get("status") == "ok":
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError("unee serve did not become healthy")

    def __exit__(self, *exc) -> None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        time.sleep(3)


def s1mb(size: str, gguf: str, run_id: str, slots: int, ctx: int, benchmarks: list[str] | None = None) -> int:
    with Server(gguf, slots=slots, ctx=ctx) as srv:
        extra = [a for b in (benchmarks or []) for a in ("--benchmark", b)]
        return run(["bench/run_s1mb.py", "--unee-url", srv.url, "--", "--data-dir", "bench/S1MB/evaluator/data", "run",
                    "--offline-dataset", "--adapter", "typesafe", "--model", f"unee-{size}", "--run-id", run_id, *extra],
                   py=str(S1MB_PY))


def main() -> None:
    s = load_state()
    for soup, parts in SOUPS.items():
        if not (ROOT / f"models/{soup}/merged").exists() and all((ROOT / f"models/{p}/merged").exists() for p in parts):
            run(["tools/soup.py", "--out", f"models/{soup}/merged", *[f"models/{p}/merged" for p in parts]])
    s.setdefault("val", {})
    for size, names in CANDIDATES.items():
        for n in names:
            if (ROOT / f"models/{n}/merged").exists() and n not in s["val"]:
                v = {k: val_acc(n, k) for k in VALS}
                v["mean"] = None if None in v.values() else sum(v.values()) / 3
                s["val"][n] = v
                save_state(s)
                log(f"{size} {n} {v}")
    s["chosen"] = {}
    for size, names in CANDIDATES.items():
        scored = {n: s["val"][n]["mean"] for n in names if s["val"].get(n, {}).get("mean") is not None}
        s["chosen"][size] = max(scored, key=scored.get)
        log(f"CHOSEN {size}: {s['chosen'][size]} ({scored})")
    save_state(s)

    s.setdefault("decidebench", {})
    for size, name in s["chosen"].items():
        if size not in s["decidebench"]:
            run(["bench/run_local.py", "--model", f"models/{name}/merged", "--device", "cuda", "--format", "compact",
                 "--name", f"unee-u4-{size}"])
            summ = ROOT / f"bench/results/unee-u4-{size}.summary.json"
            if summ.exists():
                d = json.loads(summ.read_text())
                s["decidebench"][size] = {k: d.get(k) for k in ("acc", "pair_acc", "ece")}
                save_state(s)
                log(f"DecideBench {size} {name}: {s['decidebench'][size]}")

    s.setdefault("gguf", {})
    for size, name in s["chosen"].items():
        prefix = f"models/gguf/unee-u4-{size}"
        if not (ROOT / f"{prefix}-Q4_K_M.gguf").exists():
            run([BASH, "tools/export_gguf.sh", f"models/{name}/merged", prefix], py="")
        s["gguf"][size] = f"{prefix}-Q4_K_M.gguf"
    if not (ROOT / "models/onnx-unee-u4-0.8b/onnx/model_q4f16.onnx").exists():
        run(["tools/export_onnx.py", "--merged", f"models/{s['chosen']['0.8b']}/merged", "--out",
             "models/onnx-unee-u4-0.8b", "--variant", "q4f16", "q4"])
    save_state(s)

    s.setdefault("s1mb", {})
    for size in ("0.8b", "2b"):
        if size in s["s1mb"] and s["s1mb"][size].get("complete"):
            continue
        main_dir, long_dir = S1MB_RESULTS / f"unee-u4-{size}", S1MB_RESULTS / f"unee-u4-{size}-long"
        if not main_dir.exists():
            s1mb(size, s["gguf"][size], f"unee-u4-{size}", slots=4, ctx=16384)
        todo = summarise(main_dir)["incomplete"] if main_dir.exists() else []
        if todo and not long_dir.exists():  # long documents: one slot with the whole 32k context
            s1mb(size, s["gguf"][size], f"unee-u4-{size}-long", slots=1, ctx=32768, benchmarks=todo)
        merged = summarise(main_dir, long_dir) if long_dir.exists() else summarise(main_dir)
        s["s1mb"][size] = {**merged, "long_rerun": todo}
        save_state(s)
        log(f"S1MB {size}: task avg {merged.get('task_avg')} tasks {merged.get('tasks')} complete {merged.get('complete')}")

    s.setdefault("speed", {})
    for size in ("0.8b", "2b"):
        if size in s["speed"]:
            continue
        llama = subprocess.Popen([LLAMA_CUDA, "-m", s["gguf"][size], "--jinja", "-ngl", "99", "-np", "4", "-c", "16384",
                                  "--port", "8030", "--host", "127.0.0.1", "--no-webui"],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    if httpx.get("http://127.0.0.1:8030/health", timeout=2).json().get("status") == "ok":
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            run(["bench/speed.py", "--url", "http://127.0.0.1:8030/v1", "--name", f"unee-u4-{size}-cuda", "--n", "60"])
            run(["bench/run_server.py", "--url", "http://127.0.0.1:8030/v1", "--name", f"unee-u4-{size}-q4km-cuda-c4",
                 "--format", "compact", "--concurrency", "4"])
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(llama.pid)], capture_output=True)
            time.sleep(3)
        s["speed"][size] = f"bench/results/unee-u4-{size}-cuda.speed.json"
        save_state(s)

    s.setdefault("massive", {})
    for size in ("0.8b", "2b"):
        name = f"u4-{size}"
        if name in s["massive"]:
            continue
        with Server(s["gguf"][size]) as srv:
            run(["bench/run_massive.py", "--url", srv.url, "--name", name, "--n", "60"])
        out = ROOT / f"bench/results/massive-{name}.json"
        if out.exists():
            d = json.loads(out.read_text())
            s["massive"][name] = {k: d[k] for k in ("mean", "english", "languages_within_10pt_of_english")}
            save_state(s)
            log(f"MASSIVE {name}: {s['massive'][name]}")

    rec = ROOT / f"docs/recordings/{time.strftime('%Y-%m-%d')}-unee-u4-0.8b-cpu.json"
    if not s.get("recording"):
        with Server(s["gguf"]["0.8b"], port=8021, cpu=True, slots=1) as srv:
            run(["bench/record_demo.py", "--url", srv.url, "--out", str(rec)])
        s["recording"] = str(rec.relative_to(ROOT))
        save_state(s)

    for size, name in s["chosen"].items():
        if not (ROOT / f"models/{name}/chat60.json").exists():
            run(["train/eval_chat.py", "answer", "--model", f"models/{name}/merged", "--out",
                 f"models/{name}/chat60.json", "--n", "60", "--max-tokens", "400"])
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools/restart_gen.ps1", "-NoGen",
         "-Ngl", "26"], py="")
    s.setdefault("knowledge", {})
    for size in ("0.8b", "2b"):
        name = f"u4-{size}"
        if name in s["knowledge"]:
            continue
        with Server(s["gguf"][size], port=8021, cpu=True, slots=1) as srv:
            run(["train/eval_knowledge.py", "answer", "--url", srv.url, "--name", name])
        run(["train/eval_knowledge.py", "judge", "--name", name])
        out = ROOT / f"bench/results/knowledge-{name}.json"
        if out.exists():
            d = json.loads(out.read_text())
            s["knowledge"][name] = {k: d[k] for k in ("acceptable", "answerable", "unanswerable", "english",
                                                      "other_languages")}
            save_state(s)
            log(f"knowledge {name}: {s['knowledge'][name]}")
    s.setdefault("chat", {})
    for size, name in s["chosen"].items():
        base = "base08" if size == "0.8b" else "base2b"
        out = ROOT / f"models/{name}/chat60_judge.json"
        if not out.exists():
            run(["train/eval_chat.py", "judge", "--a", f"models/{base}/chat60.json", "--b",
                 f"models/{name}/chat60.json", "--out", str(out)])
        if out.exists():
            s["chat"][size] = json.loads(out.read_text())["b_win_rate"]
            log(f"chat judge {size} {name}: win rate {s['chat'][size]}")
    save_state(s)
    log("release pass done")


if __name__ == "__main__":
    main()
