"""Self-distilled chat data: the base model answers real user prompts, so mixing these into training keeps its
text ability while it learns decisions.

    llama-server (CPU) -m models/gguf/Qwen3.5-0.8B-Q8_0.gguf --jinja -t 8 -np 2 -c 8192 --port 8090
    PYTHONUTF8=1 .venv/Scripts/python data/chat_distill.py --url http://127.0.0.1:8090/v1 --n 1200

Prompts: English first turns of OpenAssistant oasst1 (Apache-2.0), in data/generated/chat_prompts.jsonl.
Answers cut off at max_tokens are kept with "truncated": true; training then skips the end-of-turn token for them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


async def main_async(args) -> None:
    prompts = [json.loads(l)["prompt"] for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    done = {json.loads(l)["prompt"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if out.exists() else set()
    todo = [p for p in prompts[: args.n] if p not in done]
    url = args.url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    kept = cut = 0
    print(f"{len(todo)} prompts to answer ({len(done)} done)", flush=True)
    async with httpx.AsyncClient(timeout=900) as client:
        with out.open("a", encoding="utf-8") as f:
            async def one(p: str) -> None:
                nonlocal kept, cut
                body = {"messages": [{"role": "user", "content": p}], "max_tokens": args.max_tokens,
                        "temperature": 0.7, "top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}}
                async with sem:
                    try:
                        r = await client.post(url, json=body)
                        r.raise_for_status()
                        c = r.json()["choices"][0]
                    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                        print(f"error: {e}"[:200], flush=True)
                        return
                text = (c["message"].get("content") or "").strip()
                if not text:
                    return
                truncated = c.get("finish_reason") != "stop"
                cut += truncated
                f.write(json.dumps({"prompt": p, "response": text, "truncated": truncated}, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
                if (kept + cut) % 25 == 0:
                    print(f"kept={kept} cut={cut}", flush=True)

            await asyncio.gather(*(one(p) for p in todo))
    print(f"done: kept={kept} cut={cut}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--prompts", default=str(HERE / "generated" / "chat_prompts.jsonl"))
    ap.add_argument("--out", default=str(HERE / "generated" / "chat_distill.jsonl"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--concurrency", type=int, default=2)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
