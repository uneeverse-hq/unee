// Built-in retrieval for knowledge-grounded chat. A port of unee/knowledge.py that ranks identically:
// split documents into passages, rank them with BM25, put the best ones in the prompt.
// Words are runs of letters, digits and combining marks in any script; scripts written without spaces
// (Chinese, Japanese, Thai, Lao, Khmer, Myanmar) are matched on character pairs.

import { knowledgeMessages } from "./prompt.js";

const STOP = new Set(
  ("a an and are as at be by can do does for from has have how i if in is it its me my of on or our so that the " +
   "this to was we what when where which who will with you your").split(" "));
const NO_SPACE = [[0x3040, 0x30ff], [0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff], [0x0e00, 0x0eff],
  [0x1000, 0x109f], [0x1780, 0x17ff]];
const SENTENCE_END = /(?<=[.!?])\s+|(?<=[。！？])/u;

const noSpace = (text) => [...text].some((c) => {
  const o = c.codePointAt(0);
  return NO_SPACE.some(([lo, hi]) => o >= lo && o <= hi);
});

function words(text) {
  const out = [];
  for (let w of text.toLowerCase().match(/[\p{L}\p{N}\p{M}]+/gu) || []) {
    if (noSpace(w)) {
      const cs = [...w];
      for (let i = 0; i < Math.max(1, cs.length - 1); i++) out.push(cs.slice(i, i + 2).join(""));
      continue;
    }
    if (STOP.has(w)) continue;
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1); // refunds -> refund
    out.push(w);
  }
  return out;
}

// Rough word count; text without spaces counts one word per two characters.
function count(text) {
  const n = text.split(/\s+/).filter(Boolean).length;
  return noSpace(text) ? Math.max(n, Math.floor([...text].length / 2)) : n;
}

// Documents are strings or {title, text}. Passages hold up to about maxWords words, split at paragraph breaks
// (or sentence breaks inside long paragraphs), each starting with its document's title.
export function passages(docs, maxWords = 120) {
  const out = [];
  for (const d of docs) {
    const [title, text] = typeof d === "object" && d !== null ? [d.title || "", d.text || ""] : ["", String(d)];
    const units = [];
    for (const raw of text.split(/\n\s*\n/)) {
      const p = raw.trim();
      if (p) units.push(...(count(p) > maxWords ? p.split(SENTENCE_END).filter(Boolean) : [p]));
    }
    let cur = [];
    let n = 0;
    const flush = () => out.push((title ? `${title}: ` : "") + cur.join("\n"));
    for (const u of units) {
      const k = count(u);
      if (cur.length && n + k > maxWords) {
        flush();
        cur = [];
        n = 0;
      }
      cur.push(u);
      n += k;
    }
    if (cur.length) flush();
  }
  return out;
}

// The k passages that best match the query, best first; passages with no word in common are dropped.
export function search(query, chunks, k = 4) {
  if (chunks.length <= k) return [...chunks];
  const docs = chunks.map(words);
  const avg = docs.reduce((a, d) => a + d.length, 0) / docs.length || 1;
  const df = new Map();
  for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) || 0) + 1);
  const q = [...new Set(words(query))];
  const scored = [];
  docs.forEach((d, i) => {
    const tf = new Map();
    for (const w of d) tf.set(w, (tf.get(w) || 0) + 1);
    let s = 0;
    for (const w of q) {
      const f = tf.get(w);
      if (!f) continue;
      const n = df.get(w);
      s += (Math.log(1 + (docs.length - n + 0.5) / (n + 0.5)) * f * 2.2) / (f + 1.2 * (0.25 + (0.75 * d.length) / avg));
    }
    if (s > 0) scored.push([-s, i]);
  });
  scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return scored.slice(0, k).map(([, i]) => chunks[i]);
}

// Chat messages with the passages most relevant to the conversation added as knowledge. The query is the last
// two user turns, so a short follow-up ("and how much is it?") still finds its topic.
export function ground(messages, docs, k = 4) {
  const users = messages.filter((m) => m.role === "user").map((m) => m.content);
  return knowledgeMessages(messages, search(users.slice(-2).join(" "), passages(docs), k));
}
