"""Assemble the Hugging Face repos for a release into models/hf/<repo>/, ready to upload. Nothing is uploaded.

    PYTHONUTF8=1 .venv/Scripts/python tools/stage_hf.py

Reads the release pass's choices (models/release_r3.json) and makes:
- unee-0.8b: transformers weights plus the ONNX build in onnx/, so `Unee.load("uneeverse/unee-0.8b")` works in the
  browser and `from_pretrained` works in Python, and downloads count towards one repo.
- unee-2b: transformers weights.
- unee-0.8b-GGUF, unee-2b-GGUF: the 4-bit GGUF and an Ollama Modelfile.
Each gets README.md (docs/MODEL_CARD.md with per-repo front matter), LICENSE and NOTICE. Large files are hard links.

Upload (after `hf auth login`, by the owner):  hf upload uneeverse/unee-0.8b models/hf/unee-0.8b .
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORG = "uneeverse"
BASE = {"0.8b": "Qwen/Qwen3.5-0.8B", "2b": "Qwen/Qwen3.5-2B"}
WEIGHTS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def card(extra: dict, intro: str = "") -> str:
    """docs/MODEL_CARD.md with its front matter extended by `extra` (comment lines dropped)."""
    text = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    _, front, body = text.split("---\n", 2)
    lines = [ln for ln in front.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    lines += [f"{k}: {json.dumps(v) if isinstance(v, list) else v}" for k, v in extra.items()]
    if intro:
        body = re.sub(r"^(# .+\n)", lambda m: m.group(1) + "\n" + intro + "\n", body.lstrip("\n"), count=1)
    return "---\n" + "\n".join(lines) + "\n---\n" + body


def finish(repo: Path, readme: str) -> None:
    (repo / "README.md").write_text(readme, encoding="utf-8")
    for f in ("LICENSE", "NOTICE"):
        place(ROOT / f, repo / f)
    size = sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())
    print(f"{repo.name}: {sum(p.is_file() for p in repo.rglob('*'))} files, {size / 1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="u3", help="release tag: u3 = Unee 0.3, r3 = Unee 0.2")
    ap.add_argument("--state", help="release state file (default models/release_<tag>.json)")
    ap.add_argument("--out", default=str(ROOT / "models/hf"))
    args = ap.parse_args()
    state = json.loads(Path(args.state or ROOT / f"models/release_{args.tag}.json").read_text())
    out = Path(args.out)
    for size in ("0.8b", "2b"):
        merged = ROOT / f"models/{state['chosen'][size]}/merged"
        repo = out / f"unee-{size}"
        if repo.exists():
            shutil.rmtree(repo)
        for f in (*WEIGHTS, "model.safetensors"):
            place(merged / f, repo / f)
        onnx = ROOT / state.get("onnx", {}).get(size, f"models/onnx-unee-{args.tag}-{size}")
        if onnx.exists():
            for f in (onnx / "onnx").iterdir():
                place(f, repo / "onnx" / f.name)
            # transformers.js reads its settings from config.json; generation_config ends turns on <|im_end|>.
            cfg = json.loads((merged / "config.json").read_text())
            cfg["transformers.js_config"] = json.loads((onnx / "config.json").read_text())["transformers.js_config"]
            (repo / "config.json").unlink()
            (repo / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            place(onnx / "generation_config.json", repo / "generation_config.json")
            # transformers.js takes the chat template from tokenizer_config.json only, never from chat_template.jinja;
            # without it every decide() and stream() call fails. Python reads either, and the two are identical.
            tok = json.loads((merged / "tokenizer_config.json").read_text(encoding="utf-8"))
            tok["chat_template"] = (merged / "chat_template.jinja").read_text(encoding="utf-8")
            (repo / "tokenizer_config.json").unlink()
            (repo / "tokenizer_config.json").write_text(json.dumps(tok, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            place(merged / "generation_config.json", repo / "generation_config.json")
        finish(repo, card({"base_model": BASE[size], "base_model_relation": "finetune",
                           "library_name": "transformers"}))

        gguf = out / f"unee-{size}-GGUF"
        if gguf.exists():
            shutil.rmtree(gguf)
        name = f"unee-{size}-Q4_K_M.gguf"
        place(ROOT / state["gguf"][size], gguf / name)
        modelfile = (ROOT / "deploy/ollama/Modelfile").read_text(encoding="utf-8")
        modelfile = re.sub(r"uneeverse/unee\b", f"uneeverse/unee:{size}", modelfile.replace("./unee-0.8b-Q4_K_M.gguf",
                                                                                         f"./{name}")).replace("deploy/ollama/Modelfile", "Modelfile")
        (gguf / "Modelfile").write_text(modelfile, encoding="utf-8")
        intro = (f"4-bit GGUF (Q4_K_M) of [{ORG}/unee-{size}](https://huggingface.co/{ORG}/unee-{size}). "
                 f"Run it with `unee serve --model {name}` (decisions and chat), llama.cpp, or Ollama (chat).")
        finish(gguf, card({"base_model": f"{ORG}/unee-{size}", "base_model_relation": "quantized",
                           "library_name": "gguf"}, intro))


if __name__ == "__main__":
    main()
