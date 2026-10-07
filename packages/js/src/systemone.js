// /v1/systemone request parsing and answer formatting (Jev-compatible). Mirrors unee/systemone.py.

import { LETTERS } from "./prompt.js";

const NOUL_DEFAULT = {
  true: "Yes: the statement in the question holds for this input.",
  false: "No: the statement in the question does not hold for this input.",
};

export class RequestError extends Error {
  constructor(field, message) {
    super(`${field}: ${message}`);
    this.field = field;
  }
}

// Instructions and descriptions may be structured JSON, as Jev allows: compact JSON, except
// {system, instruction}, which becomes two paragraphs. Mirrors _text in unee/systemone.py.
function asText(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const keys = Object.keys(value);
    if (keys.length === 2 && keys.includes("system") && keys.includes("instruction")) {
      return `${asText(value.system)}\n\n${asText(value.instruction)}`;
    }
  }
  return JSON.stringify(value);
}

const isFilled = (v) => v && typeof v === "object" && (Array.isArray(v) ? v.length > 0 : Object.keys(v).length > 0);

function criteriaEntry(field, value) {
  if (typeof value === "string") return [value, []];
  if (value && typeof value === "object" && typeof value.what === "string") {
    if (value.examples !== undefined && !Array.isArray(value.examples)) throw new RequestError(field, "examples must be a list");
    return [value.what, value.examples || []];
  }
  if (isFilled(value)) return [asText(value), []];
  throw new RequestError(field, 'must be a string or {"what": ..., "examples": [...]}');
}

// -> {state, tasks: [{qid, kind, question, options, examples, explain}]}
export function parse(body) {
  if (!body || typeof body !== "object") throw new RequestError("body", "must be a JSON object");
  if (!("state" in body)) throw new RequestError("state", "is required");
  const qs = body.questions;
  if (!qs || typeof qs !== "object" || !Object.keys(qs).length) {
    throw new RequestError("questions", "must be a non-empty object of question_id -> question");
  }
  const tasks = [];
  for (const [qid, q] of Object.entries(qs)) {
    const where = `questions.${qid}`;
    if (!q || typeof q !== "object") throw new RequestError(where, "must be an object");
    const instructions = typeof q.instructions === "string" || isFilled(q.instructions) ? asText(q.instructions) : "";
    if (!instructions.trim()) throw new RequestError(`${where}.instructions`, "is required");
    const options = [];
    const examples = [];
    const add = (key, value, field) => {
      const [desc, ex] = criteriaEntry(field, value);
      options.push({ key, description: desc });
      for (const s of ex) examples.push([s, key]);
    };
    if (q.type === "noul") {
      const crit = q.criteria ?? NOUL_DEFAULT;
      const keys = Object.keys(crit).sort();
      if (keys.length !== 2 || keys[0] !== "false" || keys[1] !== "true") {
        throw new RequestError(`${where}.criteria`, 'noul criteria must have exactly the keys "true" and "false"');
      }
      for (const k of ["true", "false"]) add(k, crit[k], `${where}.criteria.${k}`);
    } else if (q.type === "choice") {
      if (!q.criteria || typeof q.criteria !== "object" || Array.isArray(q.criteria) || Object.keys(q.criteria).length < 2) {
        throw new RequestError(`${where}.criteria`, "choice needs at least 2 options");
      }
      for (const [k, v] of Object.entries(q.criteria)) add(String(k), v, `${where}.criteria.${k}`);
    } else if (q.type === "score") {
      if (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10) {
        throw new RequestError(`${where}.criteria`, "score needs a list of 2-10 levels, lowest first");
      }
      q.criteria.forEach((v, i) => add(String(i), v, `${where}.criteria.${i}`));
    } else {
      throw new RequestError(`${where}.type`, 'must be "noul", "choice" or "score"');
    }
    if (options.length > 255) throw new RequestError(`${where}.criteria`, "at most 255 options are supported");
    tasks.push({ qid, kind: q.type, question: instructions.trim(), options, examples, explain: Boolean(q.explain) });
  }
  return { state: body.state, tasks };
}

export function confidence(probs) {
  const n = probs.length;
  if (n < 2) return 1;
  const h = -probs.reduce((a, p) => (p > 0 ? a + p * Math.log(p) : a), 0);
  return round(Math.max(0, 1 - h / Math.log(n)));
}

const round = (x) => Math.round(x * 1e4) / 1e4;

// probs: {key: probability}
export function answer(task, probs) {
  const p = task.options.map((o) => probs[o.key]);
  const rounded = Object.fromEntries(Object.entries(probs).map(([k, v]) => [k, round(v)]));
  if (task.kind === "noul") return { type: "noul", noul: round(probs.true) };
  if (task.kind === "choice") {
    const choice = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));
    return { type: "choice", choice, probabilities: rounded, confidence: confidence(p) };
  }
  return {
    type: "score",
    score: round(p.reduce((a, v, i) => a + i * v, 0)),
    legend: Object.fromEntries(task.options.map((o) => [o.key, o.description])),
    probabilities: rounded,
    confidence: confidence(p),
  };
}
