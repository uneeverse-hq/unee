"""One command to serve Unee: starts llama.cpp's llama-server with the model, then the Unee API on top of it.

    unee serve --model unee-0.8b-Q4_K_M.gguf                  # CPU, http://127.0.0.1:8000
    unee serve --model unee-2b-Q4_K_M.gguf --gpu-layers 99    # offload to a GPU build of llama.cpp

llama-server is found from --llama-server, then $UNEE_LLAMA_SERVER, then PATH (`brew install llama.cpp`,
`winget install llama.cpp`, or a release zip from github.com/ggml-org/llama.cpp).
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

import httpx


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_llama_server(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("UNEE_LLAMA_SERVER"), "llama-server"):
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if found := shutil.which(candidate):
            return found
    sys.exit("llama-server not found: install llama.cpp, or pass --llama-server / set UNEE_LLAMA_SERVER")


def serve(args) -> None:
    import uvicorn

    from unee.server import Engine, build_app

    port = free_port()
    cmd = [find_llama_server(args.llama_server), "-m", args.model, "--jinja", "--host", "127.0.0.1",
           "--port", str(port), "-np", str(args.slots), "-c", str(args.ctx), "-ngl", str(args.gpu_layers),
           "--alias", args.model_name, "--no-webui"]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    log = open(args.llama_log, "a") if args.llama_log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + args.startup_timeout
        while True:
            if proc.poll() is not None:
                sys.exit(f"llama-server exited with code {proc.returncode} (see --llama-log)")
            try:
                if httpx.get(url + "/health", timeout=2).json().get("status") == "ok":
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.time() > deadline:
                sys.exit("llama-server did not become healthy in time")
            time.sleep(0.5)
        print(f"Unee is serving {args.model} on http://{args.host}:{args.port}  (POST /v1/systemone, "
              f"/v1/chat/completions)", flush=True)
        uvicorn.run(build_app(Engine(url, args.model_name, args.temperature)), host=args.host, port=args.port,
                    log_level="warning")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(prog="unee", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve a GGUF model with the Unee API")
    s.add_argument("--model", required=True, help="path to a Unee GGUF file")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--threads", type=int, help="CPU threads for llama.cpp (default: its own choice)")
    s.add_argument("--slots", type=int, default=4, help="requests processed in parallel")
    s.add_argument("--ctx", type=int, default=16384, help="total context across slots")
    s.add_argument("--gpu-layers", type=int, default=0, help="layers to offload to a GPU build (99 = all)")
    s.add_argument("--temperature", type=float, default=1.0, help="calibration temperature for probabilities")
    s.add_argument("--model-name", default="unee")
    s.add_argument("--llama-server", help="path to the llama-server binary")
    s.add_argument("--llama-log", help="append llama-server's output to this file")
    s.add_argument("--startup-timeout", type=float, default=180)
    args = ap.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
