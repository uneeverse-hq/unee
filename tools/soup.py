"""Model soup: average the weights of several fine-tuned models that share a base (uniform average).

    PYTHONUTF8=1 .venv/Scripts/python tools/soup.py --out models/soup-a/merged models/v1/merged models/v1r/merged

Averaging independently fine-tuned runs smooths out run-to-run noise; pick which runs to average on our own val
set (train/eval_val.py), not on DecideBench.
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="merged model directories (same architecture)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    total = None
    for m in args.models:
        files = sorted(glob.glob(str(Path(m) / "*.safetensors")))
        if len(files) != 1:
            raise SystemExit(f"{m}: expected one safetensors file, found {len(files)}")
        w = load_file(files[0])
        if total is None:
            total = {k: v.float() for k, v in w.items()}
            dtypes = {k: v.dtype for k, v in w.items()}
        else:
            if w.keys() != total.keys():
                raise SystemExit(f"{m}: tensor names differ")
            for k, v in w.items():
                total[k] += v.float()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: (v / len(args.models)).to(dtypes[k]) for k, v in total.items()}, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    for f in Path(args.models[0]).iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy(f, out / f.name)
    (out / "soup.txt").write_text("\n".join(args.models) + "\n")
    print(f"averaged {len(args.models)} models -> {out}")


if __name__ == "__main__":
    main()
