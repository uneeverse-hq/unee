// Deterministic date facts added to the prompt. Mirrors unee/facts.py; both must produce identical strings.

import { ML_MONTHS } from "./facts_months.js";

const W = "[\\p{L}\\p{N}_]"; // Python's \w for text patterns
// Python's \b is Unicode-aware; JS's is ASCII-only, so spell the boundary out.
const B = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`;
const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
const MON = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\.?";
const DAY = "(\\d{1,2})(?:st|nd|rd|th)?";
const PATTERNS = [
  ["iso", B + "((?:19|20)\\d\\d)-(\\d\\d)-(\\d\\d)" + B, "gu"],
  ["mdy", B + MON + " " + DAY + ",? ((?:19|20)\\d\\d)" + B, "gu"],
  ["dmy", B + DAY + " " + MON + ",? ((?:19|20)\\d\\d)" + B, "gu"],
  ["md", B + MON + " " + DAY + B + "(?!,? (?:19|20)\\d\\d)(?!:)", "gu"],
];

// Other languages (see ML_PATTERNS in unee/facts.py): local month names after a day number, and the CJK
// year-month-day forms. Digits may be Western, Arabic-Indic, Persian, Devanagari or Bengali.
const DIG = "[0-9\\u0660-\\u0669\\u06f0-\\u06f9\\u0966-\\u096f\\u09e6-\\u09ef]";
const DIGIT_BASES = [0x30, 0x660, 0x6f0, 0x966, 0x9e6];
const byLength = (a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0);
const ML_MON = "(" + Object.keys(ML_MONTHS).sort(byLength).join("|") + ")";
const ML_DAY = `(?<!${W})(${DIG}{1,2})(?:er)?\\.?\\s(?:de\\s)?`;
const ML_YEAR = `(?:\\s(?:de\\s|del\\s)?|,\\s?)(${DIG}{4})(?!${W})`;
const ML_PATTERNS = [
  ["ml-dmy", ML_DAY + ML_MON + "\\.?" + ML_YEAR, "giu"],
  ["ml-ymd", `(${DIG}{4})\\s?[年년]\\s?(${DIG}{1,2})\\s?[月월]\\s?(${DIG}{1,2})\\s?[日일]`, "gu"],
  ["ml-dm", ML_DAY + ML_MON + `(?!${W})(?!\\.?` + ML_YEAR + ")", "giu"],
  ["ml-md", `(?<!${W})(${DIG}{1,2})\\s?[月월]\\s?(${DIG}{1,2})\\s?[日일]`, "gu"],
];
const MAX_FACTS = 6;

function num(s) {
  let n = 0;
  for (const c of s) {
    const o = c.codePointAt(0);
    n = n * 10 + (o - DIGIT_BASES.find((b) => o >= b && o <= b + 9));
  }
  return n;
}

// Days since epoch for a valid calendar date, else null (rejects e.g. Feb 30).
function dayNumber(y, m, d) {
  if (!m) return null;
  const t = Date.UTC(y, m - 1, d);
  const x = new Date(t);
  if (x.getUTCFullYear() !== y || x.getUTCMonth() !== m - 1 || x.getUTCDate() !== d) return null;
  return t / 86400000;
}

// [day number or null, [month, day] for year-less dates or null]; day null and md null means "no date".
function parse(kind, m) {
  const inYears = (y) => y >= 1900 && y <= 2099;
  switch (kind) {
    case "iso": return [dayNumber(+m[1], +m[2], +m[3]), null];
    case "mdy": return [dayNumber(+m[3], MONTHS[m[1].slice(0, 3).toLowerCase()], +m[2]), null];
    case "dmy": return [dayNumber(+m[3], MONTHS[m[2].slice(0, 3).toLowerCase()], +m[1]), null];
    case "md": return [null, [MONTHS[m[1].slice(0, 3).toLowerCase()], +m[2]]];
    case "ml-dmy": return [inYears(num(m[3])) ? dayNumber(num(m[3]), ML_MONTHS[m[2].toLowerCase()], num(m[1])) : null, null];
    case "ml-ymd": return [inYears(num(m[1])) ? dayNumber(num(m[1]), num(m[2]), num(m[3])) : null, null];
    case "ml-dm": return [null, [ML_MONTHS[m[2].toLowerCase()], num(m[1])]];
    default: return [null, [num(m[1]), num(m[2])]]; // ml-md
  }
}

function findDates(text) {
  const found = [];
  const taken = [];
  for (const [kind, src, flags] of [...PATTERNS, ...ML_PATTERNS]) {
    for (const m of text.matchAll(new RegExp(src, flags))) {
      const start = m.index;
      const end = start + m[0].length;
      if (taken.some(([s, e]) => start < e && s < end)) continue;
      const [day, md] = parse(kind, m);
      if (md ? dayNumber(2000, md[0], md[1]) === null : day === null) continue; // 2000 is a leap year, as in Python
      taken.push([start, end]);
      found.push({ pos: start, text: m[0], day, md, year: md ? null : new Date(day * 86400000).getUTCFullYear() });
    }
  }
  return found.sort((a, b) => a.pos - b.pos);
}

export function dateFacts(text) {
  if (typeof text !== "string") return [];
  const raw = findDates(text);
  const years = raw.filter((r) => r.day !== null).map((r) => r.year);
  const resolved = [];
  for (const r of raw) {
    let day = r.day;
    if (day === null) {
      if (!years.length) continue;
      day = dayNumber(years[0], r.md[0], r.md[1]);
      if (day === null) continue;
    }
    if (resolved.every(([t]) => t !== r.text)) resolved.push([r.text, day]);
  }
  const facts = [];
  for (let i = 0; i < resolved.length; i++) {
    for (let j = i + 1; j < resolved.length; j++) {
      const [si, di] = resolved[i];
      const [sj, dj] = resolved[j];
      const n = dj - di;
      if (n === 0) facts.push(`${sj} is the same day as ${si}`);
      else facts.push(`${sj} is ${Math.abs(n)} ${Math.abs(n) === 1 ? "day" : "days"} ${n > 0 ? "after" : "before"} ${si}`);
      if (facts.length === MAX_FACTS) return facts;
    }
  }
  return facts;
}
