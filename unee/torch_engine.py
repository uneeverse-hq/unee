"""Batched GPU engine for Unee decisions, for servers with an NVIDIA GPU.

llama.cpp evaluates this hybrid model's linear-attention layers token by token, so it does not get faster with more
requests in flight. This engine runs the PyTorch model with the chunked parallel kernels (flash-linear-attention)
and dynamically batches concurrent requests into one forward pass. That forward pass returns hidden states; the
248k-row output head runs only at each prompt's last position.

    python -m unee.server --engine torch --model models/soup3/merged --port 8000
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from unee.prompt import LETTERS, build_messages


class TorchEngine:
    def __init__(self, model_path: str, model_name: str = "unee", temperature: float = 1.0, max_batch: int = 16,
                 max_wait_ms: float = 2.0, device: str = "cuda", compile: bool = False, bucket: int = 64) -> None:
        self.model_name, self.temperature, self.device = model_name, temperature, device
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
        self.backbone, self.head = self.model.model, self.model.lm_head
        self.bucket = bucket  # pad batch width to a multiple of this, so compiled shapes repeat
        if compile:
            self.backbone = torch.compile(self.backbone, dynamic=True)
        self.letter_ids = torch.tensor([self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS], device=device)
        self.stop_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.queue: list[tuple[list[int], int, Future]] = []
        self.cv = threading.Condition()
        self.gpu_lock = threading.Lock()  # decisions and generation share one GPU
        threading.Thread(target=self._loop, daemon=True).start()

    def _ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False).input_ids

    def _loop(self) -> None:
        """Collect requests for up to max_wait, then run them as one right-padded batch."""
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                self.cv.wait(timeout=self.max_wait)
                batch, self.queue = self.queue[: self.max_batch], self.queue[self.max_batch:]
            try:
                self._run(batch)
            except Exception as e:  # noqa: BLE001 - fail the requests, keep serving
                for *_, fut in batch:
                    fut.set_exception(e)

    @torch.inference_mode()
    def _run(self, batch) -> None:
        width = max(len(ids) for ids, _, _ in batch)
        width = -(-width // self.bucket) * self.bucket
        ids = torch.full((len(batch), width), self.pad, device=self.device)
        mask = torch.zeros((len(batch), width), dtype=torch.long, device=self.device)
        for i, (seq, _, _) in enumerate(batch):
            ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            mask[i, : len(seq)] = 1
        with self.gpu_lock:
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            last = hidden[torch.arange(len(batch), device=self.device), mask.sum(1) - 1]
            logits = self.head(last).float()[:, self.letter_ids]
        for i, (_, n, fut) in enumerate(batch):
            fut.set_result(torch.softmax(logits[i, :n] / self.temperature, -1).tolist())

    async def decide(self, state, task) -> tuple[dict[str, float], int]:
        ids = self._ids(build_messages(state, task.question, task.options, task.examples))
        fut: Future = Future()
        with self.cv:
            self.queue.append((ids, len(task.options), fut))
            self.cv.notify()
        probs = await asyncio.wrap_future(fut)
        return {o["key"]: p for o, p in zip(task.options, probs)}, len(ids)

    def _generate_stream(self, ids: list[int], max_new_tokens: int):
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        inputs = torch.tensor([ids], device=self.device)

        def run():
            with self.gpu_lock, torch.inference_mode():
                self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=self.stop_ids, streamer=streamer)
        threading.Thread(target=run, daemon=True).start()
        return streamer

    async def reason_stream(self, state, task, key: str):
        msgs = build_messages(state, task.question, task.options, task.examples, explain=True)
        letter = LETTERS[[o["key"] for o in task.options].index(key)]
        ids = self._ids(msgs) + self.tok(letter + "\n", add_special_tokens=False).input_ids
        async for piece in self._aiter(self._generate_stream(ids, 80)):
            yield piece

    async def reason(self, state, task, key: str) -> str:
        return "".join([p async for p in self.reason_stream(state, task, key)]).strip()

    async def chat_stream(self, messages: list[dict], max_tokens: int = 512):
        async for piece in self._aiter(self._generate_stream(self._ids(messages), max_tokens)):
            yield piece

    @staticmethod
    async def _aiter(streamer):
        """Iterate a blocking TextIteratorStreamer without blocking the event loop."""
        done = object()
        while (piece := await asyncio.to_thread(next, streamer, done)) is not done:
            if piece:
                yield piece
