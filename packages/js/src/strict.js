// Strict mode for knowledge answers: check a drafted answer against the knowledge before it is shown.
// A port of unee/strict.py that checks identically; see that file for the reasoning and the measured rates.
//
// The draft is split into sentences. Each sentence that states a fact must pass two checks:
// 1. Exact facts: numbers of three or more digits, clock times, email addresses and links must appear in the
//    knowledge or in the user's own messages.
// 2. Claim check: Unee's decision side verifies the sentence as a claim against the knowledge.
// Sentences that fail are removed. The checker is the same small model: this lowers made-up answers and cannot
// rule them out.

export const NOTE =
  "I couldn't confirm the rest from the knowledge I have. Would you like me to pass your question to a person?";
export const THRESHOLD = 0.5; // a factual sentence is kept when P(supported) is at least this

const SENTENCE_END = /(?<=[.!?])\s+|(?<=[。！？])/u;
const EMAIL = /[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/g;
const LINK = /(?:https?:\/\/|www\.)[^\s<>()"']+/gi;
const TIME = /\b([0-9]{1,2})(?::([0-9]{2}))?\s*([ap])\.?\s?m\b\.?/gi;
const NUMBER = /[0-9]+(?:[.,][0-9]+)*/g;
const THOUSANDS = /,(?=[0-9]{3}(?:[^0-9]|$))/g;

export const KIND = {
  type: "choice",
  instructions: "What kind of sentence is this, taken from a customer assistant's reply?",
  criteria: {
    fact: "It states a fact, an instruction or a policy",
    unknown: "It says the information is not known or not available",
    other: "A greeting, thanks, a question to the user, or an offer to help or pass the question on",
  },
};
export const CLAIM = {
  type: "choice",
  instructions: "Does the evidence support the claim?",
  criteria: {
    supported: "The evidence states the claim",
    contradicted: "The evidence says something different",
    not_enough_info: "The evidence does not say",
  },
};

// Sentences and list lines of a reply, in order.
export function sentences(text) {
  return text.split("\n").flatMap((line) => line.trim().split(SENTENCE_END)).map((s) => (s || "").trim()).filter(Boolean);
}

// The parts of `text` that must match the knowledge character for character: email addresses, links, clock times
// ("8 PM" and "8:00 pm" are the same time) and numbers of three or more digits ("1,250.00" and "1250" are the same
// number). Shorter numbers are left to the claim check.
export function exactFacts(text) {
  const found = new Set();
  for (const m of text.matchAll(EMAIL)) found.add(m[0].toLowerCase().replace(/[.,]+$/, ""));
  text = text.replace(EMAIL, " ");
  for (const m of text.matchAll(LINK)) {
    const link = m[0].toLowerCase().replace(/[.,;:!?]+$/, "");
    found.add(link.replace(/^(?:https?:\/\/)?(?:www\.)?/, "").replace(/\/+$/, ""));
  }
  text = text.replace(LINK, " ");
  for (const m of text.matchAll(TIME)) found.add(`${parseInt(m[1], 10)}:${m[2] || "00"}${m[3].toLowerCase()}m`);
  text = text.replace(TIME, " ");
  for (const m of text.matchAll(NUMBER)) {
    const plain = m[0].replace(THOUSANDS, "");
    if ((plain.match(/[0-9]/g) || []).length >= 3) {
      found.add(/^[0-9]+\.[0-9]+$/.test(plain) ? plain.replace(/0+$/, "").replace(/\.$/, "") : plain);
    }
  }
  return found;
}

// Exact facts in `sentence` that `source` (the knowledge plus the user's messages) does not contain.
export function unmatchedFacts(sentence, source) {
  const have = exactFacts(source);
  return [...exactFacts(sentence)].filter((f) => !have.has(f)).sort();
}

const best = (probs) => Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));

// The reply to show for `draft`, and a report of what was checked: [text, report]. `ask(items)` answers a list of
// Unee decisions, each a [state, question] pair in the /v1/systemone format, with {option: probability} for each.
export async function verify(ask, knowledge, users, draft, threshold = THRESHOLD, note = NOTE) {
  const sents = sentences(draft);
  const source = knowledge + "\n" + users.join("\n");
  const kinds = await ask(sents.map((s) => [`Sentence: ${s}`, KIND]));
  const report = sents.map((s, i) => ({ sentence: s, kind: best(kinds[i]), missing: unmatchedFacts(s, source) }));
  const facts = report.map((r, i) => i).filter((i) => report[i].kind === "fact" || report[i].missing.length);
  const claims = facts.filter((i) => !report[i].missing.length);
  const checked = await ask(claims.map((i) => [`Evidence:\n${knowledge}\n\nClaim: ${sents[i]}`, CLAIM]));
  for (const i of facts) {
    report[i].supported = report[i].missing.length ? 0 : Math.round(checked[claims.indexOf(i)].supported * 1e4) / 1e4;
  }
  const [text, action] = assemble(report, draft, threshold, note);
  return [text, { strict: action, removed: report.filter((r) => !r.kept).length, threshold, sentences: report }];
}

// The reply for checked sentences at `threshold`, and what happened to it: "sent" (nothing removed), "trimmed" or
// "declined" (no fact left). Marks every sentence kept or not. A null threshold removes nothing.
export function assemble(report, draft, threshold, note = NOTE) {
  for (const r of report) r.kept = threshold === null || !("supported" in r) || r.supported >= threshold;
  const kept = report.filter((r) => r.kept).map((r) => r.sentence);
  if (kept.length === report.length) return [draft ?? kept.join(" "), "sent"];
  if (report.some((r) => r.kept && "supported" in r)) return [[...kept, note].join(" "), "trimmed"];
  return [note, "declined"];
}
