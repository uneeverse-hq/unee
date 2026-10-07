---
# Hugging Face front matter for the published cards (one card per size; set base_model per card).
license: apache-2.0
pipeline_tag: text-generation
tags: [decision-model, classification, chatbot, rag, multilingual, on-device, webgpu, transformers.js, gguf, onnx, jev-compatible, unee, uneeverse]
# language: the MASSIVE languages where Unee 0.3 2B scored at least 60% (bench/results/massive-u3-2b.json).
language: [af, ar, az, bn, da, de, el, en, es, fa, fi, fr, he, hi, hy, id, is, it, ja, kn, ko, lv, ml, ms, my, nb, nl, pl, pt, ro, ru, sl, sv, ta, te, th, tr, ur, vi, zh]
---

# Unee 0.3

**Unee, from UNEEVERSE, by Muneef Mumthas.** A small open AI model that lives inside your app, with two jobs.

**A decision model.** It reads an input and a question with options, and returns a calibrated probability for every option in a single forward pass:
- **noul:** yes or no
- **choice:** pick one option
- **score:** rate on a scale

It can explain a decision in one sentence.

**A streaming chatbot.**
- Answers from your own documents (FAQ, policies, product pages) through built-in retrieval, and says when they don't cover the question.
- Summarises long multi-channel threads.
- Replies in the user's language.

| | Unee 0.8B | Unee 2B |
|---|---|---|
| Base | Qwen/Qwen3.5-0.8B (Apache-2.0) | Qwen/Qwen3.5-2B (Apache-2.0) |
| Parameters | 0.75B | 1.9B |
| Files | GGUF Q4_K_M 529 MB; ONNX q4f16 469 MB (WebGPU) | GGUF Q4_K_M 1.27 GB |
| DecideBench v1.1, bf16 | 84.0% (pair 70.0%, ECE 0.025) | 88.0% (pair 76.5%, ECE 0.026) |
| DecideBench v1.1, 4-bit GGUF | 83.0% | 87.0% |
| S1MB Task Avg (official / clean, 101 benchmarks with no training overlap) | 35.45 / 36.74 (#3 under 1B) | 44.41 / 45.69 (best 2B-class) |
| MASSIVE, 51 languages: English / average / 12 priority languages | 91.7% / 63.3% / 74.4% | 86.7% / 69.7% / 77.2% |
| Answers from your own documents: good / correct "I don't know" (116 questions, 9B judge) | 38.8% / 53.6% | 48.3% / 67.9% |
| Replies with a made-up fact: no check / strict mode (same 116 questions and drafts) | 14.7% / 11.2% | 10.3% / 6.0% |
| General benchmarks (lm-eval, bf16): MMLU 5-shot / GSM8K / IFEval per prompt, strict | 49.3 / 29.7 / 39.7 | 58.1 / 49.0 / 52.7 |
| One decision, laptop RTX 4070, 4-bit (p50) | 96 ms | 111 ms |
| Browser (WebGPU) | yes, 0.4–0.9 s per decision on a laptop GPU | not packaged |

Version 0.3, 2026-10-07. The 0.8B model is a 50/50 weight average of Unee 0.2 0.8B and a run continued from it on public decision data; the 2B model is a continued run. Training data: our teacher-labelled synthetic tasks; the train splits of [bekko-system-one-dataset-v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) from permissively licensed sources only, with every S1MB test input excluded; SQuAD 2.0 (CC BY-SA 4.0) for grounded answers. A leakage check (`tools/contamination_check.py`) finds no overlap with DecideBench; S1MB overlap is reported with a clean score above.

## Quick start

**In the browser or Node.js** (`npm i unee @huggingface/transformers`):

```js
import { Unee } from "unee";
const unee = await Unee.load("uneeverse/unee-0.8b", { device: "webgpu" }); // or { device: "cpu" } in Node
const res = await unee.decide({ state: "My card was charged twice", questions: {
  team: { type: "choice", instructions: "Which team handles this?", criteria: { billing: "Charges and refunds", technical: "Bugs" } } } });
for await (const piece of unee.stream([{ role: "user", content: "Do you ship to Ireland?" }], { knowledge: ["Shipping: UK only."] })) console.log(piece);
```

**As a server** (`pip install unee`, with llama.cpp's `llama-server` on the PATH): `unee serve --model unee-0.8b-Q4_K_M.gguf`. It serves the Jev-compatible `POST /v1/systemone` and an OpenAI-compatible `POST /v1/chat/completions` (streaming, with an optional `knowledge` field).

**Strict mode** (`"strict": true` with `knowledge`, or `strict: true` in JS): Unee writes the whole answer, then its decision side checks every sentence against your documents and removes what it cannot support, so the answer arrives checked and whole. It lowers made-up facts (numbers in the table above); it does not rule them out.

**Open Knowledge Format bundles** (OKF v0.2, a folder of Markdown files with a small YAML header each) can be the knowledge: `unee.okf.load_bundle("help-centre/")` in Python, `okfDocuments({ "refunds.md": text, ... })` in JS.

**Chat in Ollama:** `ollama run uneeverse/unee` (decisions need `unee serve`).

**Prompt format matters.** Decisions use Unee's compact prompt plus the date-facts step. Use the `unee` packages, or copy `unee/prompt.py` and `unee/facts.py` from [the repository](https://github.com/uneeverse-hq/unee).

## How it was made

- **Teacher.** Qwen3.5-9B (Q4_K_M, open weights, Apache-2.0) writes decision tasks and labels them. On DecideBench the teacher itself scores 96.75%.
  - The tasks span 40 families (agent tool routing, agent action approval, claim checking, policy moderation, date and amount policies, triage, intent, sentiment, PII, prompt injection, and more) across 48 domains.
  - Each task comes with contrastive pairs: one small edit changes the right answer.
  - The teacher's probabilities are the labels; items where its top answer is below 0.6 are dropped.
- **Student.** LoRA (r=32, all linear layers), 2 epochs. The loss is soft-target cross-entropy over the option letters; options and examples are shuffled.
  - 15% of samples have no worked examples, and 15% use the "explain" format.
  - **Round 2:** 10% of samples were self-distilled chat answers to OpenAssistant (oasst1, Apache-2.0) prompts, to keep chat ability. A blind teacher judgement against the base model gave a 43.8% win rate over 40 comparisons, within noise of no change.
  - **Round 3:** 15% of samples are teacher-written chat, with answers up to 512 tokens. That covers:
    - general answers (the teacher is asked to be concise, so the student learns short answers)
    - knowledge-grounded answers through the same retrieval and prompt as serving, including "I don't know" when the documents don't cover a question (unanswerable answers that weren't refusals were dropped)
    - summaries of email, chat and call threads
  - **Multilingual (round 3).** About 30% of the chat data and a translated share of the decision data are in other languages. 45 languages, with Arabic, Hindi, Spanish, Chinese, French, Portuguese, Bengali, Russian, Urdu, Indonesian, German and Japanese weighted first.
    - Decision items are translated by the teacher, keeping names, numbers and dates verbatim; items that lose one are dropped.
    - Qwen3.5, the base, was trained on 201 languages and dialects.
- **Model soup.** The released 0.8B averages two training runs. The released 2B is a single run.
- **Choice (0.3).** By a rule fixed before any result: the best mean of three held-out validation sets (English, translated, and S1MB-style from public validation splits), among the previous release, the new run and their 50/50 weight average. No benchmark took part. Re-scored without the selection rows that overlap training text, both picks are unchanged.
- **Date facts.** Before the model runs, a deterministic preprocessor adds the day differences between any dates in the input (`unee/facts.py`, `packages/js/src/facts.js`). It's part of the prompt format, so use the provided libraries or replicate it.

## Evaluation notes

- **DecideBench test items and worked examples were never used for training.** They carry a canary string.
- **DecideBench was used during development** to compare candidate approaches, so the numbers aren't from a fully untouched test set.
  - The 0.8B soup recipe was chosen on our own held-out set.
  - Souping all 2B runs was decided before results were seen.
- **Prompt format.** DecideBench numbers use Unee's compact prompt, with one worked example per option passed the way the benchmark passes them.
- **General benchmarks** (EleutherAI lm-evaluation-harness 0.4.13, bf16, thinking off, greedy): ARC-Challenge normalised 43.5 / 49.9, HellaSwag normalised 51.3 / 59.6, TruthfulQA mc1 32.4 / 29.9 and mc2 49.3 / 45.8, IFEval per instruction 51.2 / 62.6 (0.8B / 2B). Qwen's own cards give IFEval 52.1 / 61.2 for the bases, sampled and without naming the IFEval score used, so these are not like-for-like; we did not run the bases ourselves.
- **Same result through the API.** Through the HTTP server, the benchmark's own Jev-style adapter gives the same accuracy as the direct runner. That was measured with an earlier 0.8B: 76.5% both ways.

## Limitations

- **Single forward pass, no reasoning.** Multi-step rules, arithmetic other than date differences, and long inputs are weaker. Unee scores below the teacher and the large hosted models.
- **Date facts only cover explicit dates** (ISO, "May 1, 2026", "1 May 2026", "May 1" when a year appears elsewhere). Relative dates ("three weeks ago") aren't resolved.
- **Quality varies by language.** English is strongest. See the per-language MASSIVE results (51 languages, the same utterances in each) before relying on Unee in a given language. Below 60% for the 2B: Amharic, Welsh, Hungarian, Javanese, Georgian, Khmer, Mongolian, Albanian, Swahili and Tagalog.
- **Open-ended chat is not its strength.** On 60 general questions a blind Qwen3.5-9B judge preferred the base Qwen3.5 answers (Unee won 35% for 0.8B, 21% for 2B). Unee answers are about half as long and can contain factual mistakes. Use its chat with your own documents.
- **Answers from your documents need a person checking.** About half are judged fully correct, and 10-15% of replies contain a fact the documents do not state (6-11% with strict mode). The test is 116 questions with a model as judge, and its results move by a few points between runs.
- **The streamed reason is the model's explanation, not proof of how it decided.**
- **Not for high-stakes decisions** (medical, legal, credit, employment) without human review.

## License

Apache-2.0 (weights and code), copyright Muneef Mumthas / UNEEVERSE. Redistributions keep the NOTICE file. If you build with Unee, please credit "Unee by UNEEVERSE". The base models, teacher and oasst1 prompts are Apache-2.0.
