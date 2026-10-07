"""S1MB (System One Mosaic Benchmark) against a running Unee server, through S1MB's own TypeSafe (Jev) adapter.

    unee serve --model models/gguf/unee-0.8b-Q4_K_M.gguf --gpu-layers 99 --port 8000
    bench/S1MB/.venv/Scripts/python bench/run_s1mb.py --unee-url http://127.0.0.1:8000 -- \
        run --adapter typesafe --model unee-0.8b --run-id unee-08b-20261005 [--category smoke-v1 --limit 2]

Unee's /v1/systemone accepts Jev requests unchanged, structured instructions included. So the only differences
from S1MB's Jev run are the base URL, no API key, and Unee's 4-decimal rounding; S1MB's prompts, scoring and
validation are untouched. Everything after `--` goes to S1MB's own CLI. S1MB lives in bench/S1MB (git-ignored)
with its own venv:
    git clone --depth 1 https://github.com/hotchpotch/S1MB.git bench/S1MB
    py -3.11 -m venv bench/S1MB/.venv && bench/S1MB/.venv/Scripts/python -m pip install -e bench/S1MB/evaluator
"""

from __future__ import annotations

import argparse
import os
import sys

from s1mb import cli
from s1mb.adapters import typesafe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unee-url", default="http://127.0.0.1:8000")
    args, rest = ap.parse_known_args()

    class UneeAdapter(typesafe.TypeSafeAdapter):
        provider = "unee"
        base_url = args.unee_url
        rounding_decimals = 4  # Unee rounds probabilities to 4 decimals

    os.environ.setdefault("TYPESAFE_API_KEY", "local")  # the adapter insists on a key; Unee ignores it
    typesafe.TypeSafeAdapter = UneeAdapter  # S1MB's CLI imports the class at run time
    sys.argv = [sys.argv[0], *[a for a in rest if a != "--"]]
    cli.main()


if __name__ == "__main__":
    main()
