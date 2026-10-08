"""Install Unee the way a new user would, in places where nothing is installed yet, and check that it works.

    python tools/fresh_install_test.py                  # test the packages built in dist/ and packages/js
    python tools/fresh_install_test.py --published      # test what is on PyPI and npm instead

It makes a brand-new Python environment and a brand-new Node folder in a temporary directory, installs the Unee
packages into them and nothing else, then:

  Python   starts `unee serve` and asks it for decisions (yes/no, pick one, score), a decision with a reason,
           streaming chat, chat from documents, the same with strict mode, and a summary
  Node.js  loads the browser model on the processor, asks for a decision and streams a short reply

Every check prints PASS or FAIL with what came back. The exit code is 0 only when all of them pass.

The model files are taken from this machine by default (--gguf, --onnx). With --from-hub they are downloaded from
Hugging Face, as a new user would get them, which only works once the model repositories are public (or with a login).
llama.cpp is the one thing a new user installs separately (`winget install llama.cpp`); point --llama-server at it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ->  {detail}" if detail else ""), flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


CLIENT_CHECKS = textwrap.dedent('''
    import json, sys
    from unee import Client
    unee = Client(sys.argv[1], timeout=300)
    ticket = "Hi, I was charged twice for order #4821 this morning. Please send the extra $39 back to my card."
    docs = ["Refunds: refunds are paid within 14 days of a return. Shipping: UK only, 2 to 3 working days."]
    out = {}
    out["noul"] = unee.noul(ticket, "Does the customer ask for money back?")
    out["choice"] = unee.choice(ticket, "Which team should handle this ticket?",
                                {"billing": "Charges, refunds and payments", "technical": "Bugs and outages", "shipping": "Deliveries"})
    out["score"] = unee.score(ticket, "How urgent is this?", ["Can wait a week", "Within a few days", "Today"])
    out["explain"] = unee.choice(ticket, "Which team should handle this ticket?",
                                 {"billing": "Charges, refunds and payments", "technical": "Bugs and outages"}, explain=True)
    out["chat"] = "".join(unee.chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=24))
    out["knowledge"] = "".join(unee.chat([{"role": "user", "content": "How long do refunds take?"}], knowledge=docs, max_tokens=60))
    out["strict"] = "".join(unee.chat([{"role": "user", "content": "How long do refunds take?"}], knowledge=docs, strict=True, max_tokens=60))
    out["strict_report"] = unee.strict_report
    out["summary"] = "".join(unee.summarize("Ana wrote on Monday that her order 4821 was charged twice. Support refunded 39 dollars on Tuesday. She has not confirmed yet."))
    print("RESULT " + json.dumps(out, default=str))
''')

NODE_CHECKS = textwrap.dedent('''
    import { env } from "@huggingface/transformers";
    import { Unee } from "@uneeverse/unee";
    import path from "node:path";
    const source = process.argv[2];
    let id = source;
    if (!source.includes("/") || path.isAbsolute(source)) {
      const dir = path.resolve(source);
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.localModelPath = path.dirname(dir) + path.sep;
      id = path.basename(dir);
    }
    const unee = await Unee.load(id, { device: "cpu", dtype: "q4f16" });
    const decision = await unee.decide({
      state: "Hi, I was charged twice for order #4821 this morning. Please send the extra $39 back to my card.",
      questions: {
        team: { type: "choice", instructions: "Which team should handle this ticket?",
                criteria: { billing: "Charges, refunds and payments", technical: "Bugs and outages", shipping: "Deliveries" } },
        refund: { type: "noul", instructions: "Does the customer ask for money back?" },
      },
    });
    let text = "";
    for await (const piece of unee.stream([{ role: "user", content: "Say hello in five words." }], { max_new_tokens: 20, temperature: 0 })) text += piece;
    console.log("RESULT " + JSON.stringify({ decision, text }));
''')


def python_side(work: Path, args) -> None:
    print("\nPython: a new environment, `pip install unee`, `unee serve`", flush=True)
    venv = work / "venv"
    run([sys.executable, "-m", "venv", str(venv)], check=True)
    bindir = venv / ("Scripts" if WINDOWS else "bin")
    python = bindir / ("python.exe" if WINDOWS else "python")
    before = run([str(python), "-m", "pip", "list", "--format=json"]).stdout
    check("the environment starts empty", "unee" not in before, f"{len(json.loads(before))} packages (pip itself)")

    target = "unee" if args.published else str(next((ROOT / "dist").glob("unee-*.whl")))
    done = run([str(python), "-m", "pip", "install", "--quiet", target])
    version = run([str(python), "-c", "import importlib.metadata as m; print(m.version('unee'))"]).stdout.strip()
    check("pip install unee" + (" (from PyPI)" if args.published else " (the built wheel)"), done.returncode == 0 and bool(version),
          f"unee {version}" if version else done.stderr[-300:])
    if done.returncode:
        return
    cli = run([str(bindir / "unee"), "--help"])
    check("the `unee` command is there", cli.returncode == 0 and "serve" in cli.stdout)

    gguf = Path(args.gguf)
    if args.from_hub:
        run([str(python), "-m", "pip", "install", "--quiet", "huggingface_hub"], check=True)
        got = run([str(bindir / "hf"), "download", "uneeverse/unee-0.8b-GGUF", "unee-0.8b-Q4_K_M.gguf", "--local-dir", str(work)])
        gguf = work / "unee-0.8b-Q4_K_M.gguf"
        check("the model downloads from Hugging Face", got.returncode == 0 and gguf.exists(), "" if gguf.exists() else got.stderr[-300:])
        if not gguf.exists():
            return

    port = "8177"
    serve = [str(bindir / "unee"), "serve", "--model", str(gguf), "--port", port, "--llama-log", str(work / "llama.log")]
    if args.llama_server:
        serve += ["--llama-server", args.llama_server]
    log = open(work / "serve.log", "w", encoding="utf-8")
    server = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT, cwd=work)
    try:
        ready = False
        for _ in range(180):
            if server.poll() is not None:
                break
            probe = run([str(python), "-c", f"import httpx; print(httpx.get('http://127.0.0.1:{port}/health', timeout=2).status_code)"])
            if probe.stdout.strip() == "200":
                ready = True
                break
            time.sleep(1)
        check("`unee serve` starts and answers /health", ready, "" if ready else (work / "serve.log").read_text(errors="replace")[-400:])
        if not ready:
            return
        script = work / "client_checks.py"
        script.write_text(CLIENT_CHECKS, encoding="utf-8")
        got = run([str(python), str(script), f"http://127.0.0.1:{port}"], cwd=work)
        line = next((l for l in got.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            check("the Python client talks to the server", False, (got.stderr or got.stdout)[-500:])
            return
        out = json.loads(line[7:])
        check("yes or no", isinstance(out["noul"], (int, float)) and 0 <= out["noul"] <= 1, f"{out['noul']:.3f} that the customer asks for money back")
        choice = out["choice"]
        check("pick one", choice.get("choice") == "billing" and abs(sum(choice["probabilities"].values()) - 1) < 0.01,
              f"{choice.get('choice')} at {max(choice['probabilities'].values()):.3f}")
        check("score on a scale", 0 <= out["score"]["score"] <= 2, f"{out['score']['score']:.2f} of 2")
        check("a decision with its reason", bool(out["explain"].get("reason")), out["explain"].get("reason", "")[:90])
        check("streaming chat", len(out["chat"].strip()) > 0, out["chat"].strip()[:90])
        check("chat from documents", "14" in out["knowledge"], out["knowledge"].strip()[:110])
        report = out["strict_report"] or {}
        check("strict mode", bool(out["strict"].strip()) and "strict" in report, f"{report.get('strict')}: {out['strict'].strip()[:90]}")
        check("summary", len(out["summary"].strip()) > 0, out["summary"].strip().replace("\n", " ")[:110])
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        if WINDOWS:
            run(["taskkill", "/F", "/T", "/PID", str(server.pid)])


def node_side(work: Path, args) -> None:
    print("\nNode.js: a new folder, `npm i @uneeverse/unee`, the browser model on the processor", flush=True)
    folder = work / "node"
    folder.mkdir()
    npm = shutil.which("npm")
    if not npm:
        check("npm is installed", False, "install Node.js to run this half")
        return
    run([npm, "init", "-y"], cwd=folder, check=True)
    if args.published:
        target = "@uneeverse/unee"
    else:
        packed = run([npm, "pack", "--silent", "--pack-destination", str(work)], cwd=ROOT / "packages" / "js")
        target = str(work / packed.stdout.strip().splitlines()[-1])
    done = run([npm, "install", "--silent", target, "@huggingface/transformers"], cwd=folder)
    installed = (folder / "node_modules" / "@uneeverse" / "unee" / "package.json").exists()
    check("npm i @uneeverse/unee" + (" (from npm)" if args.published else " (the packed tarball)"), done.returncode == 0 and installed,
          json.loads((folder / "node_modules" / "@uneeverse" / "unee" / "package.json").read_text())["version"] if installed else done.stderr[-300:])
    if not installed:
        return
    (folder / "checks.mjs").write_text(NODE_CHECKS, encoding="utf-8")
    source = "uneeverse/unee-0.8b" if args.from_hub else args.onnx
    got = run([shutil.which("node"), "checks.mjs", source], cwd=folder)
    line = next((l for l in got.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        check("the model loads and answers in Node.js", False, (got.stderr or got.stdout)[-500:])
        return
    out = json.loads(line[7:])
    answers = out["decision"]["answers"]
    check("a decision in Node.js", answers["team"]["choice"] == "billing" and 0 <= answers["refund"]["noul"] <= 1,
          f"team {answers['team']['choice']}, asks for money back {answers['refund']['noul']}")
    check("streaming chat in Node.js", len(out["text"].strip()) > 0, out["text"].strip()[:90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", action="store_true", help="install from PyPI and npm instead of the local builds")
    ap.add_argument("--from-hub", action="store_true", help="download the model files from Hugging Face")
    ap.add_argument("--gguf", default=str(ROOT / "models" / "hf" / "unee-0.8b-GGUF" / "unee-0.8b-Q4_K_M.gguf"))
    ap.add_argument("--onnx", default=str(ROOT / "models" / "hf" / "unee-0.8b"))
    ap.add_argument("--llama-server", default=None, help="path to llama-server, if it is not on PATH")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder to look inside")
    ap.add_argument("--only", choices=["python", "node"], default=None)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="unee-fresh-"))
    print(f"Working in {work} (nothing from this repository is on its path)")
    try:
        if args.only != "node":
            python_side(work, args)
        if args.only != "python":
            node_side(work, args)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed." + (f" Failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
