"""Strict mode for knowledge answers: check a drafted answer against the knowledge before it is sent.

The draft is split into sentences, and each sentence that states a fact must pass two checks:
1. **Exact facts.** Numbers of three or more digits, clock times, email addresses and links must appear in the
   knowledge or in the user's own messages. An invented phone number or price fails here, with no model call.
2. **Claim check.** Unee's decision side verifies the sentence as a claim against the knowledge (supported,
   contradicted or not enough information), all sentences of a reply in one batch of forward passes.

Sentences that state no fact (a greeting, a question to the user, an offer to pass the question on, "I don't know")
are kept. Sentences that fail are removed; if any were removed, a short note says the rest could not be confirmed,
and if no fact is left the reply is only that note.

The checker is the same small model, so this lowers made-up answers and cannot rule them out. The measured rates
are in bench/results/knowledge-*.json (train/eval_knowledge.py). Mirrored in packages/js/src/strict.js.
"""

from __future__ import annotations

import re

NOTE = "I couldn't confirm the rest from the knowledge I have. Would you like me to pass your question to a person?"
THRESHOLD = 0.5  # a factual sentence is kept when P(supported) is at least this

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")
# ASCII classes on purpose, so the JavaScript port matches character for character.
_EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_LINK = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+", re.I)
_TIME = re.compile(r"\b([0-9]{1,2})(?::([0-9]{2}))?\s*([ap])\.?\s?m\b\.?", re.I)
_NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_THOUSANDS = re.compile(r",(?=[0-9]{3}(?:[^0-9]|$))")

KIND = {"type": "choice", "instructions": "What kind of sentence is this, taken from a customer assistant's reply?",
        "criteria": {"fact": "It states a fact, an instruction or a policy",
                     "unknown": "It says the information is not known or not available",
                     "other": "A greeting, thanks, a question to the user, or an offer to help or pass the question on"}}
CLAIM = {"type": "choice", "instructions": "Does the evidence support the claim?",
         "criteria": {"supported": "The evidence states the claim",
                      "contradicted": "The evidence says something different",
                      "not_enough_info": "The evidence does not say"}}


def sentences(text: str) -> list[str]:
    """Sentences and list lines of a reply, in order."""
    return [s.strip() for line in text.split("\n") for s in _SENTENCE_END.split(line.strip()) if s and s.strip()]


def exact_facts(text: str) -> set[str]:
    """The parts of `text` that must match the knowledge character for character: email addresses, links, clock
    times ("8 PM" and "8:00 pm" are the same time) and numbers of three or more digits ("1,250.00" and "1250" are
    the same number). Shorter numbers are left to the claim check: "two weeks" and "14 days" both occur."""
    found = {m.group().lower().rstrip(".,") for m in _EMAIL.finditer(text)}
    text = _EMAIL.sub(" ", text)
    for m in _LINK.finditer(text):
        link = m.group().lower().rstrip(".,;:!?")
        found.add(re.sub(r"^(?:https?://)?(?:www\.)?", "", link).rstrip("/"))
    text = _LINK.sub(" ", text)
    found |= {f"{int(m.group(1))}:{m.group(2) or '00'}{m.group(3).lower()}m" for m in _TIME.finditer(text)}
    text = _TIME.sub(" ", text)
    for m in _NUMBER.finditer(text):
        plain = _THOUSANDS.sub("", m.group())
        if len(re.findall(r"[0-9]", plain)) >= 3:
            found.add(plain.rstrip("0").rstrip(".") if re.fullmatch(r"[0-9]+\.[0-9]+", plain) else plain)
    return found


def unmatched_facts(sentence: str, source: str) -> list[str]:
    """Exact facts in `sentence` that `source` (the knowledge plus the user's messages) does not contain."""
    return sorted(exact_facts(sentence) - exact_facts(source))


async def verify(ask, knowledge: str, users: list[str], draft: str, threshold: float = THRESHOLD,
                 note: str = NOTE) -> tuple[str, dict]:
    """The reply to send for `draft`, and a report of what was checked. `ask(items)` answers a list of Unee
    decisions, each a (state, question) pair in the /v1/systemone format, with {option: probability} for each."""
    sents = sentences(draft)
    source = knowledge + "\n" + "\n".join(users)
    kinds = await ask([(f"Sentence: {s}", KIND) for s in sents])
    report = [{"sentence": s, "kind": max(k, key=k.get), "missing": unmatched_facts(s, source)}
              for s, k in zip(sents, kinds)]
    facts = [i for i, r in enumerate(report) if r["kind"] == "fact" or r["missing"]]
    claims = [i for i in facts if not report[i]["missing"]]
    checked = dict(zip(claims, await ask([(f"Evidence:\n{knowledge}\n\nClaim: {sents[i]}", CLAIM) for i in claims])))
    for i in facts:
        report[i]["supported"] = 0.0 if report[i]["missing"] else round(checked[i]["supported"], 4)
    text, action = assemble(report, draft, threshold, note)
    return text, {"strict": action, "removed": sum(not r["kept"] for r in report), "threshold": threshold,
                  "sentences": report}


def assemble(report: list[dict], draft: str | None, threshold: float | None, note: str = NOTE) -> tuple[str, str]:
    """The reply for checked sentences at `threshold`, and what happened to it: "sent" (nothing removed), "trimmed"
    or "declined" (no fact left). Marks every sentence "kept" or not. A threshold of None removes nothing, which
    gives the unchecked draft; `draft` keeps the original line breaks when nothing is removed."""
    for r in report:
        r["kept"] = threshold is None or "supported" not in r or r["supported"] >= threshold
    kept = [r["sentence"] for r in report if r["kept"]]
    if len(kept) == len(report):
        return (draft if draft is not None else " ".join(kept)), "sent"
    if any(r["kept"] and "supported" in r for r in report):
        return " ".join(kept + [note]), "trimmed"
    return note, "declined"
