"""LoRA fine-tune of a small decoder into a decision model, on data/generated/train.jsonl.

    PYTHONUTF8=1 .venv/Scripts/python train/train.py --out models/unee-v1 --epochs 1

Each training item uses the compact serving prompt (unee.prompt.build_messages: question, options, one solved
example per option, then the state), with options and examples shuffled so letters and order carry no prior. The loss is the cross-entropy of the full-vocabulary
next-token distribution against a soft target spread over the option letters: (1 - alpha) * gold + alpha * teacher.
A share of items use the "explain" prompt, where the letter is followed by a one-sentence reason trained with the
normal language-model loss, so the model can stream its reason after deciding. With --chat, a share of samples are
chat answers (data/chat_teacher.py: general, knowledge-grounded and summary turns written by the teacher; or the
older self-distilled data/chat_distill.py) trained with the language-model loss, so the model chats well too.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from unee.prompt import LETTERS, build_messages  # noqa: E402
from unee.systemone import NOUL_DEFAULT  # noqa: E402


def encode(tok, rec: dict, rng: random.Random, p_zero_shot: float, p_explain: float, max_len: int,
           p_noul_default: float = 0.0):
    """One training sequence in the compact serving format, with options and examples in random order.
    With p_noul_default, a yes/no item is shown exactly as the API builds it when the caller gives no criteria:
    the default descriptions, "true" first."""
    opts = list(rec["options"])
    if {o["key"] for o in opts} == {"true", "false"} and rng.random() < p_noul_default:
        opts = [{"key": k, "description": NOUL_DEFAULT[k]} for k in ("true", "false")]
    else:
        rng.shuffle(opts)
    keys = [o["key"] for o in opts]
    shots = [(e["state"], e["gold"]) for e in rec["examples"]] if rng.random() >= p_zero_shot else []
    rng.shuffle(shots)
    explain = bool(rng.random() < p_explain and rec.get("why"))
    msgs = build_messages(rec["state"], rec["question"], opts, shots, explain=explain)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, add_special_tokens=False).input_ids
    decision_pos = len(ids) - 1  # the logits here predict the answer letter
    tail: list[int] = []
    if explain:
        tail = tok("\n" + rec["why"].strip() + "<|im_end|>", add_special_tokens=False).input_ids
        ids = ids + tok(LETTERS[keys.index(rec["gold"])], add_special_tokens=False).input_ids + tail
    if len(ids) > max_len:
        return None
    return {"ids": ids, "decision_pos": decision_pos, "keys": keys, "gold": rec["gold"],
            "teacher": rec.get("teacher") or {}, "n_tail": len(tail), "tail_weight": 0.5}


def encode_chat(tok, rec: dict, max_len: int, max_tail: int = 128):
    """A chat turn (`messages`, or a single user `prompt`); only the answer tokens carry loss. The answer is cut to
    `max_tail` tokens."""
    msgs = rec.get("messages") or [{"role": "user", "content": rec["prompt"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, add_special_tokens=False).input_ids
    end = "" if rec.get("truncated") else "<|im_end|>"  # a cut-off answer must not teach stopping there
    tail = tok(rec["response"] + end, add_special_tokens=False).input_ids
    tail = tail[:max_tail]  # a cut tail simply ends without the end-of-turn token
    if len(ids) + len(tail) > max_len:
        return None
    return {"ids": ids + tail, "decision_pos": len(ids) - 1, "keys": [], "gold": None, "teacher": {},
            "n_tail": len(tail), "tail_weight": 1.0}


def tail_loss(head, hidden: torch.Tensor, target: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """Mean next-token cross-entropy over answer tokens. Each token needs a full-vocabulary (248k) logit row, so
    the rows are made chunk by chunk and recomputed in the backward pass instead of kept: long chat answers then
    cost about as much memory as short ones."""
    from torch.utils.checkpoint import checkpoint
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, len(target), chunk):
        total = total + checkpoint(lambda h, t: torch.nn.functional.cross_entropy(head(h).float(), t, reduction="sum"),
                                   hidden[s:s + chunk], target[s:s + chunk], use_reentrant=False)
    return total / len(target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--accum", type=int, default=16, help="samples per optimizer step")
    ap.add_argument("--micro", type=int, default=4, help="samples per forward pass")
    ap.add_argument("--random-batches", action="store_true",
                    help="mix lengths (and so task families) inside each batch instead of grouping by length")
    ap.add_argument("--alpha-teacher", type=float, default=0.3, help="weight of teacher probs in the soft target")
    ap.add_argument("--teacher-temp", type=float, default=1.0,
                    help="soften teacher probs (p^(1/T)); T>1 passes on more of the runner-up options")
    ap.add_argument("--p-zero-shot", type=float, default=0.15)
    ap.add_argument("--p-explain", type=float, default=0.15)
    ap.add_argument("--p-noul-default", type=float, default=0.0,
                    help="share of yes/no items shown with the API's default criteria (callers often give none)")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--chat", help="self-distilled chat jsonl (prompt, response) to mix in")
    ap.add_argument("--p-chat", type=float, default=0.1, help="share of samples drawn from --chat")
    ap.add_argument("--chat-max-tail", type=int, default=128, help="chat answers are cut to this many tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: load the frozen base in 4-bit (merge happens on CPU)")
    ap.add_argument("--resume-adapter", help="continue an interrupted run from its saved adapter (<out>/adapter)")
    ap.add_argument("--resume-seen", type=int, default=0, help="samples the saved adapter had already seen")
    args = ap.parse_args()

    # A resumed run draws a fresh sample order, so it does not replay the start of the interrupted one.
    rng = random.Random(f"{args.seed}-{args.resume_seen}" if args.resume_seen else args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    # split on "\n" only: str.splitlines() also breaks on U+2028 and similar characters inside JSON strings
    recs = [json.loads(l) for l in Path(args.train).read_text(encoding="utf-8").split("\n") if l.strip()]
    chats = ([json.loads(l) for l in Path(args.chat).read_text(encoding="utf-8").split("\n") if l.strip()]
             if args.chat else [])
    print(f"{len(chats)} chat samples mixed in at p={args.p_chat if chats else 0}", flush=True)
    steps_total = math.ceil(len(recs) * args.epochs)
    print(f"{len(recs)} training items, {steps_total} samples over {args.epochs} epochs", flush=True)

    if args.load_4bit:  # QLoRA: 4-bit frozen base, so bigger students fit next to other GPU users
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, quantization_config=quant,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules="all-linear", exclude_modules=r".*(visual|lm_head|mtp).*")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if args.resume_adapter:  # the LoRA weights come back; the optimizer's moments restart from zero
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        loaded = set_peft_model_state_dict(model, load_file(str(Path(args.resume_adapter) / "adapter_model.safetensors")))
        if loaded.unexpected_keys:
            raise SystemExit(f"adapter does not match the model: {loaded.unexpected_keys[:3]}")
        print(f"resumed from {args.resume_adapter} at {args.resume_seen} samples", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    updates = math.ceil(steps_total / args.accum)
    warm = max(1, updates // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / max(1, updates)))))
    for _ in range(args.resume_seen // args.accum):  # the learning-rate schedule continues where the run stopped
        sched.step()

    # The backbone returns hidden states; the (tied, 248k-row) head runs only where a loss is taken, which keeps
    # memory small enough for micro-batches of whole few-shot prompts.
    backbone = model.base_model.model.model
    head = model.base_model.model.lm_head
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batches():
        order: list[int] = []
        while True:
            while len(order) < args.micro * 16:
                more = list(range(len(recs)))
                rng.shuffle(more)
                order += more
            pool = [encode_chat(tok, rng.choice(chats), args.max_len, args.chat_max_tail)
                    if chats and rng.random() < args.p_chat
                    else encode(tok, recs[order.pop()], rng, args.p_zero_shot, args.p_explain, args.max_len,
                                args.p_noul_default)
                    for _ in range(args.micro * 16)]
            yield len([p for p in pool if p is None])
            pool = [p for p in pool if p is not None]
            if not args.random_batches:  # length-sorted batches pad less but group similar items together
                pool.sort(key=lambda e: len(e["ids"]))
            chunks = [pool[i:i + args.micro] for i in range(0, len(pool), args.micro)]
            rng.shuffle(chunks)
            yield from chunks

    model.train()
    seen, skipped, micro_steps = args.resume_seen, 0, 0
    t0 = time.perf_counter()
    run_loss = run_acc = run_n = 0.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    next_log = (seen // (args.accum * 10) + 1) * args.accum * 10
    next_save = (seen // 2000 + 1) * 2000

    def save_adapter() -> None:  # progress.json says how far the saved weights got, for --resume-seen
        model.save_pretrained(out / "adapter")
        (out / "adapter" / "progress.json").write_text(json.dumps({"seen": seen, "steps_total": steps_total}) + "\n")
    for batch in batches():
        if isinstance(batch, int):
            skipped += batch
            seen += batch
            continue
        if seen >= steps_total:
            break
        width = max(len(e["ids"]) for e in batch)
        ids = torch.full((len(batch), width), pad, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for i, e in enumerate(batch):
            ids[i, : len(e["ids"])] = torch.tensor(e["ids"], device="cuda")
            mask[i, : len(e["ids"])] = 1
        hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        rows = torch.arange(len(batch), device="cuda")
        logp = torch.log_softmax(head(hidden[rows, [e["decision_pos"] for e in batch]]).float(), -1)
        loss = torch.zeros((), device="cuda")
        n_decisions = 0
        for i, e in enumerate(batch):
            n = len(e["keys"])
            if e["n_tail"]:
                end = len(e["ids"])
                loss = loss + e["tail_weight"] * tail_loss(head, hidden[i, end - e["n_tail"] - 1: end - 1],
                                                           ids[i, end - e["n_tail"]: end])
            if not n:
                continue
            n_decisions += 1
            target = torch.zeros(n, device="cuda")
            target[e["keys"].index(e["gold"])] = 1 - args.alpha_teacher if e["teacher"] else 1.0
            if e["teacher"]:  # teacher probabilities, softened by --teacher-temp (1 = as given)
                t = torch.tensor([max(e["teacher"].get(k, 0.0), 1e-6) for k in e["keys"]], device="cuda")
                t = t ** (1 / args.teacher_temp)
                target += args.alpha_teacher * t / t.sum()
            target = target / target.sum()
            lp = logp[i, letter_ids[:n]]
            loss = loss - (target * lp).sum()
            run_acc += float(lp.argmax().item() == e["keys"].index(e["gold"]))
        loss = loss / len(batch)
        (loss * len(batch) / args.accum).backward()
        run_loss += loss.item() * len(batch)
        run_n += n_decisions
        seen += len(batch)
        micro_steps += len(batch)
        if micro_steps >= args.accum:
            micro_steps = 0
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if seen >= next_log:
            next_log += args.accum * 10
            dt = time.perf_counter() - t0
            print(f"[{seen}/{steps_total}] loss={run_loss / max(run_n, 1):.4f} acc={run_acc / max(run_n, 1):.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {(seen - args.resume_seen) / dt:.2f} samples/s skipped={skipped} "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            run_loss = run_acc = run_n = 0.0
        if seen >= next_save:
            next_save += 2000
            save_adapter()
    save_adapter()
    if args.load_4bit:  # merge the adapter into a full-precision copy of the base, on the CPU
        from peft import PeftModel
        del model, opt
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    else:
        merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1) + "\n")
    print(f"done: {seen} samples, {skipped} skipped, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
