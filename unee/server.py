"""Unee HTTP API on top of llama-server.

    llama-server -m unee.gguf --jinja -np 4 -c 16384 --port 8080      # CPU or GPU build of llama.cpp
    python -m unee.server --llama http://127.0.0.1:8080 --port 8000

Endpoints:
  POST /v1/systemone         Jev-compatible decisions (noul, choice, score; many questions per call). A question
                             with "explain": true also gets a one-sentence "reason". With "stream": true the response
                             is SSE: a `decision` event with every answer first, then `reason` deltas, then `done`.
  POST /v1/chat/completions  OpenAI-compatible chat, streamed or not (passed through to llama-server). Extension:
                             "knowledge": [doc, ...] (strings or {"title", "text"}) makes Unee answer from those
                             documents, retrieving the most relevant passages per turn (unee/knowledge.py).
                             "strict": true (or {"threshold": 0.5, "note": "..."}) checks the answer sentence by
                             sentence against the knowledge before sending it (unee/strict.py); the reply then
                             arrives whole, with the check's report under "unee".
  GET  /health
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import time

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from unee import __version__
from unee.knowledge import ground
from unee.systemone import RequestError, Task, answer, parse, state_value
from unee.prompt import LETTERS, build_messages, knowledge_of
from unee.strict import NOTE, THRESHOLD, verify

NO_THINK = {"enable_thinking": False}


class Engine:
    def __init__(self, llama_url: str, model_name: str, temperature: float = 1.0) -> None:
        self.url = llama_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model_name
        self.temperature = temperature  # calibration temperature, fitted on held-out data (train/eval_val.py)
        self.client = httpx.AsyncClient(timeout=600)

    async def decide(self, state, task: Task) -> tuple[dict[str, float], int]:
        """Probability of each option from one prompt pass (the answer letter's logprobs)."""
        body = {"messages": build_messages(state, task.question, task.options, task.examples), "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": NO_THINK}
        r = await self.client.post(self.url, json=body)
        r.raise_for_status()
        data = r.json()
        seen: dict[str, float] = {}
        for t in data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            seen[t["token"].strip()] = seen.get(t["token"].strip(), 0.0) + math.exp(t["logprob"])
        floor = min(seen.values(), default=1e-6) * 0.5
        raw = {o["key"]: seen.get(LETTERS[i], floor) ** (1 / self.temperature) for i, o in enumerate(task.options)}
        z = sum(raw.values())
        return {k: v / z for k, v in raw.items()}, data.get("usage", {}).get("prompt_tokens", 0)

    def reason_body(self, state, task: Task, key: str, stream: bool) -> dict:
        """The explain prompt with the decided letter prefilled; llama-server echoes the prefill in its output."""
        msgs = build_messages(state, task.question, task.options, task.examples, explain=True)
        letter = LETTERS[[o["key"] for o in task.options].index(key)]
        msgs.append({"role": "assistant", "content": letter + "\n"})
        return {"messages": msgs, "max_tokens": 80, "temperature": 0, "stream": stream, "chat_template_kwargs": NO_THINK}

    async def reason(self, state, task: Task, key: str) -> str:
        r = await self.client.post(self.url, json=self.reason_body(state, task, key, False))
        r.raise_for_status()
        return strip_letter(r.json()["choices"][0]["message"]["content"])

    async def reason_stream(self, state, task: Task, key: str):
        """Reason deltas, with the leading letter line held back and dropped."""
        head, past_letter = "", False
        async with self.client.stream("POST", self.url, json=self.reason_body(state, task, key, True)) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                if past_letter:
                    yield delta
                    continue
                head += delta
                if "\n" in head or len(head) > 3:
                    past_letter = True
                    if rest := strip_letter(head):
                        yield rest


def strip_letter(text: str) -> str:
    """Drop a first line that is just an option letter ("A", "A.", "(A)")."""
    first, _, rest = text.lstrip().partition("\n")
    letter = first.strip(" .()")
    return rest.strip() if len(letter) == 1 and letter in LETTERS else text.strip()


GROUP = 8  # options per round; a question with more options is decided as a tournament


async def decide_any(engine, state, task: Task) -> tuple[dict[str, float], int]:
    """Decide among any number of options. Up to GROUP at once; beyond that, each group of GROUP sends its best
    option to the next round. An eliminated option's probability is its share within its group times the final
    probability of that group's winner, so every option still gets a probability."""
    if len(task.options) <= GROUP:
        return await engine.decide(state, task)
    groups = [task.options[i:i + GROUP] for i in range(0, len(task.options), GROUP)]

    def sub(opts):
        keys = {o["key"] for o in opts}
        return dataclasses.replace(task, options=opts, examples=[e for e in task.examples if e[1] in keys])

    results = await asyncio.gather(*(engine.decide(state, sub(g)) for g in groups))
    winners = [max(p, key=p.get) for p, _ in results]
    final, n_final = await decide_any(engine, state, sub([o for o in task.options if o["key"] in winners]))
    probs = {}
    for (p, _), w in zip(results, winners):
        for k, v in p.items():
            probs[k] = final[w] * v / p[w] if k != w else final[w]
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()}, sum(n for _, n in results) + n_final


def build_app(engine: Engine) -> Starlette:
    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "model": engine.model_name, "version": __version__})

    async def systemone(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": {"field": "body", "message": "invalid JSON"}}, status_code=422)
        try:
            state, tasks = parse(body)
        except RequestError as e:
            return JSONResponse({"error": {"field": e.field, "message": e.message}}, status_code=422)
        state = state_value(state)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(decide_any(engine, state, t) for t in tasks))
        answers = {t.qid: answer(t, probs) for t, (probs, _) in zip(tasks, results)}
        usage = {"input_tokens": sum(n for _, n in results), "output_tokens": 0}
        decided = {t.qid: max(p, key=p.get) for t, (p, _) in zip(tasks, results)}
        explained = [t for t in tasks if t.explain]
        payload = {"model": engine.model_name, "answers": answers, "usage": usage,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if not body.get("stream"):
            reasons = await asyncio.gather(*(engine.reason(state, t, decided[t.qid]) for t in explained))
            for t, text in zip(explained, reasons):
                answers[t.qid]["reason"] = text
            return JSONResponse(payload)

        async def events():
            yield f"event: decision\ndata: {json.dumps(payload)}\n\n"
            for t in explained:
                async for delta in engine.reason_stream(state, t, decided[t.qid]):
                    yield f"event: reason\ndata: {json.dumps({'question': t.qid, 'delta': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def ask(items: list[tuple]) -> list[dict[str, float]]:
        """Option probabilities for a list of (state, question) decisions, for strict mode's checks."""
        parsed = [parse({"state": s, "questions": {"q": q}}) for s, q in items]
        results = await asyncio.gather(*(decide_any(engine, state_value(s), tasks[0]) for s, tasks in parsed))
        return [probs for probs, _ in results]

    async def strict_chat(body: dict, knowledge: str, options) -> Response:
        """Write the whole answer, check it sentence by sentence (unee/strict.py), then send what passed."""
        if hasattr(engine, "chat_stream"):
            draft = "".join([p async for p in engine.chat_stream(body["messages"], body.get("max_tokens") or 512)])
        else:
            r = await engine.client.post(engine.url, json={**body, "stream": False, "chat_template_kwargs": NO_THINK})
            r.raise_for_status()
            draft = r.json()["choices"][0]["message"].get("content") or ""
        opts = options if isinstance(options, dict) else {}
        users = [m["content"] for m in body["messages"] if m["role"] == "user"]
        text, report = await verify(ask, knowledge, users, draft.strip(), float(opts.get("threshold", THRESHOLD)),
                                    opts.get("note", NOTE))
        if not body.get("stream"):
            return JSONResponse({"object": "chat.completion", "model": engine.model_name, "unee": report, "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]})

        async def checked():
            chunk = {"object": "chat.completion.chunk", "model": engine.model_name, "unee": report,
                     "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(checked(), media_type="text/event-stream")

    async def chat(request: Request) -> Response:
        body = await request.json()
        docs = body.pop("knowledge", None)
        if docs:
            body["messages"] = ground(body["messages"], docs, int(body.pop("knowledge_k", 4)))
        strict = body.pop("strict", False)
        if strict and (knowledge := knowledge_of(body["messages"])):
            return await strict_chat(body, knowledge, strict)
        if hasattr(engine, "chat_stream"):  # the local torch engine generates itself
            pieces = engine.chat_stream(body["messages"], body.get("max_tokens") or 512)
            if not body.get("stream"):
                text = "".join([p async for p in pieces])
                return JSONResponse({"object": "chat.completion", "model": engine.model_name, "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]})

            async def sse():
                async for p in pieces:
                    chunk = {"object": "chat.completion.chunk", "model": engine.model_name,
                             "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")
        body.setdefault("chat_template_kwargs", NO_THINK)
        if not body.get("stream"):
            r = await engine.client.post(engine.url, json=body)
            return Response(r.content, status_code=r.status_code, media_type="application/json")

        async def passthrough():
            async with engine.client.stream("POST", engine.url, json=body) as r:
                async for chunk in r.aiter_raw():
                    yield chunk

        return StreamingResponse(passthrough(), media_type="text/event-stream")

    return Starlette(routes=[Route("/health", health), Route("/v1/systemone", systemone, methods=["POST"]),
                             Route("/v1/chat/completions", chat, methods=["POST"])])


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="llama", choices=("llama", "torch"),
                    help="llama: talk to a llama-server (CPU or GPU); torch: batched PyTorch engine (NVIDIA GPU)")
    ap.add_argument("--llama", default="http://127.0.0.1:8080", help="llama-server base URL")
    ap.add_argument("--model", help="Hugging Face model directory, for --engine torch")
    ap.add_argument("--max-batch", type=int, default=16, help="--engine torch: requests per forward pass")
    ap.add_argument("--compile", action="store_true", help="--engine torch: torch.compile the model (slower start)")
    ap.add_argument("--model-name", default="unee")
    ap.add_argument("--temperature", type=float, default=1.0, help="calibration temperature for option probabilities")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.engine == "torch":
        from unee.torch_engine import TorchEngine
        engine = TorchEngine(args.model, args.model_name, args.temperature, max_batch=args.max_batch,
                             compile=args.compile)
    else:
        engine = Engine(args.llama, args.model_name, args.temperature)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
