"""Round-3 release pass, run once training has finished (models/chain_release.ps1 starts it).

    PYTHONUTF8=1 .venv/Scripts/python tools/release_r3.py

Selection rule, fixed before any round-3 result existed: each size's release model is the candidate with the best
mean of English val accuracy (val.jsonl) and multilingual val accuracy (val_ml.jsonl, translated, never trained
on). DecideBench, MASSIVE and S1MB run only on the chosen models (plus base/old releases for MASSIVE context), so no
benchmark takes part in choosing.

Steps:
1. Build the soups.
2. Run val and val_ml for every candidate.
3. Choose.
4. Run DecideBench (compact).
5. Export GGUF and, for 0.8B, ONNX.
6. Run MASSIVE (51 languages) through the API for base, old and new models.
7. Run the full S1MB for the new models.
8. Answer chat60 for the chosen models, then judge them against base with the teacher.

Each finished step is recorded in models/release_r3.json, so a rerun skips it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
S1MB_PY = ROOT / "bench/S1MB/.venv/Scripts/python.exe"
BASH = r"C:\Program Files\Git\bin\bash.exe"
STATE = ROOT / "models/release_r3.json"
SOUPS = {
    # v6-08b was dropped after the 2026-10-06 restart (decided before any round-3 result existed).
    "soup-r3": ["v4-08b", "v5-08b"],
    "soup-r3x": ["v4-08b", "v5-08b", "soup4r"],
    "soup2b-r3": ["e2d", "e2e"],
    "soup2b-r3x": ["e2d", "e2e", "soup2b3"],
}
CANDIDATES = {
    # Amended 2026-10-06 07:20, after the val scores and before any other result: soup4r (round 2) won the
    # decision-only rule by 0.6 points but has none of the round-3 chat training v0.2 ships, so it is not a candidate.
    "0.8b": ["v4-08b", "v5-08b", "soup-r3", "soup-r3x"],
    "2b": ["soup2b3", "e2d", "e2e", "soup2b-r3", "soup2b-r3x"],
}
BASE = {"0.8b": ("Qwen/Qwen3.5-0.8B", "models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"),
        "2b": ("Qwen/Qwen3.5-2B", None)}
OLD_GGUF = {"0.8b": "models/gguf/unee-0.8b-Q4_K_M.gguf", "2b": "models/gguf/unee-2b-Q4_K_M.gguf"}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "models/release_r3.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], py: str = PY) -> int:
    log("run " + " ".join(args))
    with open(ROOT / "models/release_r3.log", "a", encoding="utf-8") as f:
        return subprocess.run([py, *args] if py else args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1) + "\n")


def val_acc(name: str, which: str) -> float | None:
    out = ROOT / f"models/{name}/{which}.json"
    if not out.exists():
        val = "data/generated/val.jsonl" if which == "val" else "data/generated/val_ml.jsonl"
        run(["train/eval_val.py", "--model", f"models/{name}/merged", "--val", val, "--out", str(out)])
    try:
        return json.loads(out.read_text())["t1"]["acc"]
    except (OSError, KeyError, ValueError):
        return None


class Server:
    """`unee serve` on the GPU build of llama.cpp; the whole process tree is killed on exit."""

    def __init__(self, gguf: str, port: int = 8020, cpu: bool = False) -> None:
        self.url = f"http://127.0.0.1:{port}"
        build = ["tools/llama.cpp/cpu/llama-server.exe", "--threads", "4", "--slots", "1"] if cpu else \
            ["tools/llama.cpp/cuda/llama-server.exe", "--gpu-layers", "99", "--slots", "4"]
        self.proc = subprocess.Popen(
            [PY, "-m", "unee.cli", "serve", "--model", gguf, "--llama-server", *build, "--ctx", "16384",
             "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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


def main() -> None:
    s = load_state()
    for soup, parts in SOUPS.items():
        if not (ROOT / f"models/{soup}/merged").exists() and all((ROOT / f"models/{p}/merged").exists() for p in parts):
            run(["tools/soup.py", "--out", f"models/{soup}/merged", *[f"models/{p}/merged" for p in parts]])
    s.setdefault("val", {})
    for size, names in CANDIDATES.items():
        for n in names:
            if (ROOT / f"models/{n}/merged").exists() and n not in s["val"]:
                v, m = val_acc(n, "val"), val_acc(n, "val_ml")
                s["val"][n] = {"val": v, "val_ml": m, "mean": None if None in (v, m) else (v + m) / 2}
                save_state(s)
                log(f"{size} {n} {s['val'][n]}")
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
                 "--name", f"unee-r3-{size}"])
            summ = ROOT / f"bench/results/unee-r3-{size}.summary.json"
            if summ.exists():
                d = json.loads(summ.read_text())
                s["decidebench"][size] = {k: d.get(k) for k in ("acc", "pair_acc", "ece")}
                save_state(s)
                log(f"DecideBench {size} {name}: {s['decidebench'][size]}")

    s.setdefault("gguf", {})
    for size, name in s["chosen"].items():
        prefix = f"models/gguf/unee-r3-{size}"
        if not (ROOT / f"{prefix}-Q4_K_M.gguf").exists():
            run([BASH, "tools/export_gguf.sh", f"models/{name}/merged", prefix], py="")
        s["gguf"][size] = f"{prefix}-Q4_K_M.gguf"
    if not (ROOT / "models/onnx-unee-r3-0.8b/onnx/model_q4f16.onnx").exists():
        run(["tools/export_onnx.py", "--merged", f"models/{s['chosen']['0.8b']}/merged", "--out",
             "models/onnx-unee-r3-0.8b", "--variant", "q4f16", "q4"])
    save_state(s)

    s.setdefault("massive", {})
    jobs = [("r3-0.8b", s["gguf"]["0.8b"]), ("r3-2b", s["gguf"]["2b"]), ("old-0.8b", OLD_GGUF["0.8b"]),
            ("base-0.8b", BASE["0.8b"][1]), ("old-2b", OLD_GGUF["2b"])]  # new models first, in case time runs short
    for name, gguf in jobs:
        if name in s["massive"] or not gguf or not (ROOT / gguf).exists():
            continue
        with Server(gguf) as srv:
            run(["bench/run_massive.py", "--url", srv.url, "--name", name, "--n", "60"])
        out = ROOT / f"bench/results/massive-{name}.json"
        if out.exists():
            d = json.loads(out.read_text())
            s["massive"][name] = {k: d[k] for k in ("mean", "english", "languages_within_10pt_of_english")}
            save_state(s)
            log(f"MASSIVE {name}: {s['massive'][name]}")

    s.setdefault("speed", {})
    for size in ("0.8b", "2b"):  # GPU latency for the site's speed table (same harness as before: bench/speed.py)
        if size in s["speed"]:
            continue
        llama = subprocess.Popen([str(ROOT / "tools/llama.cpp/cuda/llama-server.exe"), "-m", s["gguf"][size], "--jinja", "-ngl", "99",
                                  "-np", "4", "-c", "16384", "--port", "8030", "--host", "127.0.0.1", "--no-webui"],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    if httpx.get("http://127.0.0.1:8030/health", timeout=2).json().get("status") == "ok":
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            run(["bench/speed.py", "--url", "http://127.0.0.1:8030/v1", "--name", f"unee-r3-{size}-cuda", "--n", "60"])
            # The same 4-bit server on DecideBench, so the cost chart pairs cost with the accuracy of the priced system.
            run(["bench/run_server.py", "--url", "http://127.0.0.1:8030/v1", "--name", f"unee-r3-{size}-q4km-cuda-c4",
                 "--format", "compact", "--concurrency", "4"])
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(llama.pid)], capture_output=True)
            time.sleep(3)
        s["speed"][size] = f"bench/results/unee-r3-{size}-cuda.speed.json"
        save_state(s)

    s.setdefault("s1mb", {})
    for size in ("0.8b", "2b"):
        run_id = f"unee-r3-{size}"
        if size in s["s1mb"]:
            continue
        with Server(s["gguf"][size]) as srv:
            code = run(["bench/run_s1mb.py", "--unee-url", srv.url, "--", "--data-dir", "bench/S1MB/evaluator/data",
                        "run", "--offline-dataset", "--adapter", "typesafe", "--model", f"unee-{size}",
                        "--run-id", run_id], py=str(S1MB_PY))
        results = ROOT / f"bench/S1MB/evaluator/data/results/{run_id}"
        sys.path.insert(0, str(ROOT / "bench"))
        from s1mb_summary import summarise
        s["s1mb"][size] = {"exit": code, "results": str(results.relative_to(ROOT)),
                           **(summarise(results) if results.exists() else {})}
        save_state(s)
        log(f"S1MB {size}: exit {code}, task avg {s['s1mb'][size].get('task_avg')}")

    rec = ROOT / f"docs/recordings/{time.strftime('%Y-%m-%d')}-unee-r3-0.8b-cpu.json"
    if not rec.exists():  # the website reel re-recorded with the new 0.8B, on the CPU like the page says
        with Server(s["gguf"]["0.8b"], port=8021, cpu=True) as srv:
            run(["bench/record_demo.py", "--url", srv.url, "--out", str(rec)])
    s["recording"] = str(rec.relative_to(ROOT))
    save_state(s)

    for size, name in s["chosen"].items():
        if not (ROOT / f"models/{name}/chat60.json").exists():
            run(["train/eval_chat.py", "answer", "--model", f"models/{name}/merged", "--out",
                 f"models/{name}/chat60.json", "--n", "60", "--max-tokens", "400"])
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools/restart_gen.ps1", "-NoGen",
         "-Ngl", "26"], py="")
    # "Unee Max" question from the owner: does the 9B teacher, untouched but with Unee's compact prompt and date facts,
    # clear Jev's 98.0%? No training involved; a GPU-server option only, not potato-friendly.
    if "max_9b" not in s:
        run(["bench/run_server.py", "--name", "unee-max-9b-compact", "--format", "compact", "--url",
             "http://127.0.0.1:8080/v1", "--concurrency", "4"])
        summ = ROOT / "bench/results/unee-max-9b-compact.summary.json"
        if summ.exists():
            d = json.loads(summ.read_text())
            s["max_9b"] = {k: d.get(k) for k in ("acc", "pair_acc", "ece")}
            save_state(s)
            log(f"Unee Max (9B, compact + facts) DecideBench: {s['max_9b']}")
    # Knowledge-grounded answers (the website-assistant case), held-out companies, about 30% non-English.
    # Students answer on the CPU so the teacher keeps the GPU for writing the set and judging.
    if not (ROOT / "data/generated/kb_eval.jsonl").exists():
        run(["train/eval_knowledge.py", "make", "--n", "24"])
    s.setdefault("knowledge", {})
    for name, gguf in [("base-0.8b", BASE["0.8b"][1]), ("old-0.8b", OLD_GGUF["0.8b"]), ("r3-0.8b", s["gguf"]["0.8b"]),
                       ("old-2b", OLD_GGUF["2b"]), ("r3-2b", s["gguf"]["2b"])]:
        if name in s["knowledge"] or not gguf or not (ROOT / gguf).exists():
            continue
        with Server(gguf, port=8021, cpu=True) as srv:
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
        if not out.exists() and (ROOT / f"models/{base}/chat60.json").exists():
            run(["train/eval_chat.py", "judge", "--a", f"models/{base}/chat60.json", "--b",
                 f"models/{name}/chat60.json", "--out", str(out)])
        if out.exists():
            s["chat"][size] = json.loads(out.read_text())["b_win_rate"]
    save_state(s)
    log("release pass done")


if __name__ == "__main__":
    main()
