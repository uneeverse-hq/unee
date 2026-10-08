// Unee in the browser or Node, on transformers.js (WebGPU, with a WASM fallback). No server and no API key.
//
//   import { Unee } from "@uneeverse/unee";
//   const unee = await Unee.load("<hf-repo-or-local-path>", { device: "webgpu" });
//   const res = await unee.decide({ state: "...", questions: { spam: { type: "noul", instructions: "Is this spam?" } } });
//   for await (const piece of unee.stream([{ role: "user", content: "Write a haiku about tea." }])) print(piece);
//   for await (const piece of unee.stream(messages, { knowledge: [faqText, { title: "Refunds", text: "..." }] })) ...
//   for await (const piece of unee.summarize(longThread)) ...

import { AutoModelForCausalLM, AutoTokenizer, Tensor, TextStreamer } from "@huggingface/transformers";

import { answer, parse, RequestError } from "./systemone.js";
import { buildMessages, knowledgeOf, LETTERS, summaryMessages } from "./prompt.js";
import { ground } from "./knowledge.js";
import { NOTE, THRESHOLD, verify } from "./strict.js";

export { RequestError } from "./systemone.js";
export { buildMessages, knowledgeMessages, summaryMessages } from "./prompt.js";
export { ground, passages, search } from "./knowledge.js";
export { exactFacts, verify } from "./strict.js";
export { frontMatter, okfDocuments } from "./okf.js";

function softmax(xs, temperature) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp((x - m) / temperature));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

export class Unee {
  constructor(tokenizer, model, { temperature = 1.0 } = {}) {
    this.tokenizer = tokenizer;
    this.model = model;
    this.temperature = temperature; // calibration temperature for the option probabilities
    this.letterIds = [...LETTERS].map((c) => tokenizer.encode(c, { add_special_tokens: false })[0]);
    // Stop at the end of the assistant turn; the shipped generation config only lists <|endoftext|>.
    this.stopIds = ["<|im_end|>", "<|endoftext|>"].map((t) => tokenizer.encode(t, { add_special_tokens: false })[0]);
  }

  // The 4-bit graphs need WebGPU: onnxruntime-web's WASM build has no GatherBlockQuantized kernel.
  static async load(modelId, { device = "webgpu", dtype = "q4f16", temperature, progress_callback } = {}) {
    const tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback });
    const model = await AutoModelForCausalLM.from_pretrained(modelId, { device, dtype, progress_callback });
    return new Unee(tokenizer, model, { temperature });
  }

  #inputs(messages) {
    return this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
      enable_thinking: false,
    });
  }

  // Probability of each option for one decision task: one forward pass, logits for the last position only.
  async probabilities(state, task) {
    const inputs = this.#inputs(buildMessages(state, task.question, task.options, task.examples));
    const out = await this.model({ ...inputs, num_logits_to_keep: new Tensor("int64", [1n], []) });
    const logits = out.logits.data; // [1, 1, vocab]
    const scores = task.options.map((_, i) => Number(logits[this.letterIds[i]]));
    for (const t of Object.values(out)) t?.dispose?.(); // the recurrent/KV state tensors may live on the GPU
    const p = softmax(scores, this.temperature);
    return Object.fromEntries(task.options.map((o, i) => [o.key, p[i]]));
  }

  // A Jev /v1/systemone request -> response. Questions run one after another (one model, one device).
  // Any number of options: up to GROUP at once, otherwise a tournament (each group's best goes to the next round).
  // Mirrors decide_any in unee/server.py.
  async probabilitiesAny(state, task, GROUP = 8) {
    if (task.options.length <= GROUP) return this.probabilities(state, task);
    const sub = (opts) => {
      const keys = new Set(opts.map((o) => o.key));
      return { ...task, options: opts, examples: task.examples.filter(([, g]) => keys.has(g)) };
    };
    const groups = [];
    for (let i = 0; i < task.options.length; i += GROUP) groups.push(task.options.slice(i, i + GROUP));
    const results = [];
    for (const g of groups) results.push(await this.probabilities(state, sub(g)));
    const winners = results.map((p) => Object.keys(p).reduce((a, b) => (p[b] > p[a] ? b : a)));
    const final = await this.probabilitiesAny(state, sub(task.options.filter((o) => winners.includes(o.key))), GROUP);
    const probs = {};
    results.forEach((p, i) => {
      const w = winners[i];
      for (const [k, v] of Object.entries(p)) probs[k] = k === w ? final[w] : (final[w] * v) / p[w];
    });
    const z = Object.values(probs).reduce((a, b) => a + b, 0);
    return Object.fromEntries(Object.entries(probs).map(([k, v]) => [k, v / z]));
  }

  async decide(body) {
    const { state, tasks } = parse(body);
    const t0 = performance.now();
    const answers = {};
    for (const task of tasks) {
      const probs = await this.probabilitiesAny(state, task);
      answers[task.qid] = answer(task, probs);
      if (task.explain) {
        const key = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));
        answers[task.qid].reason = await this.reason(state, task, key);
      }
    }
    return { model: "unee", answers, latency_ms: Math.round(performance.now() - t0) };
  }

  // One sentence naming the deciding detail, with the decided letter prefilled.
  async reason(state, task, key, onToken) {
    const msgs = buildMessages(state, task.question, task.options, task.examples, true);
    const letter = LETTERS[task.options.findIndex((o) => o.key === key)];
    const prompt =
      this.tokenizer.apply_chat_template(msgs, { add_generation_prompt: true, tokenize: false, enable_thinking: false }) +
      letter + "\n";
    const inputs = this.tokenizer(prompt, { add_special_tokens: false });
    let text = "";
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (piece) => {
        text += piece;
        onToken?.(piece);
      },
    });
    await this.model.generate({ ...inputs, max_new_tokens: 80, do_sample: false, eos_token_id: this.stopIds, streamer });
    return text.trim();
  }

  // Streaming chat: yields text pieces as they are generated. With `knowledge` (documents: strings or
  // {title, text}), the most relevant passages go into the prompt and Unee answers from them.
  // With `strict` (true, or { threshold, note }) the answer is written in full, checked sentence by sentence
  // against the knowledge (strict.js), and yielded once, whole; `onReport` receives what the check did.
  async *stream(messages, { max_new_tokens = 512, temperature = 0.7, knowledge = null, knowledgeK = 4,
                            strict = false, onReport = null } = {}) {
    if (knowledge?.length) messages = ground(messages, knowledge, knowledgeK);
    const known = strict ? knowledgeOf(messages) : null;
    if (known === null) {
      yield* this.#generate(messages, max_new_tokens, temperature);
      return;
    }
    let draft = "";
    for await (const piece of this.#generate(messages, max_new_tokens, temperature)) draft += piece;
    const ask = async (items) => {
      const out = [];
      for (const [state, q] of items) {
        const parsed = parse({ state, questions: { q } });
        out.push(await this.probabilitiesAny(parsed.state, parsed.tasks[0]));
      }
      return out;
    };
    const users = messages.filter((m) => m.role === "user").map((m) => m.content);
    const [text, report] = await verify(ask, known, users, draft.trim(), strict.threshold ?? THRESHOLD, strict.note ?? NOTE);
    onReport?.(report);
    yield text;
  }

  async *#generate(messages, max_new_tokens, temperature) {
    const queue = [];
    let done = false;
    let wake = null;
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (piece) => {
        queue.push(piece);
        wake?.();
      },
    });
    const run = this.model
      .generate({ ...this.#inputs(messages), max_new_tokens, do_sample: temperature > 0, temperature,
                  eos_token_id: this.stopIds, streamer })
      .finally(() => {
        done = true;
        wake?.();
      });
    while (!done || queue.length) {
      if (queue.length) yield queue.shift();
      else await new Promise((r) => (wake = r));
    }
    await run;
  }

  // A short bullet summary of a long thread, transcript or notes, streamed. `focus` steers it, e.g.
  // "next steps for the agent".
  summarize(text, { focus = null, max_new_tokens = 400 } = {}) {
    return this.stream(summaryMessages(text, focus), { max_new_tokens, temperature: 0 });
  }
}
