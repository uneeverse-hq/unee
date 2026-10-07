"""Rebuild the website's decision reel (uneeverse-web src/content/unee.ts) from a recording made by bench/record_demo.py,
so every probability on the page is copied by a script, not by hand.

    PYTHONUTF8=1 .venv/Scripts/python tools/reel_from_recording.py docs/recordings/<file>.json \
        --web ../../../uneeverse-web/src/content/unee.ts --examples support support_ar injection agent review

Prints each answer first, so wrong ones can be left out with --examples or --skip <example>.<question>. The
recording keeps every answer either way. Options are listed most likely first.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TEAMS = {"billing": "Billing", "technical": "Technical", "shipping": "Shipping", "sales": "Sales"}
TITLES = {"support": "Support inbox", "support_ar": "Support inbox, in Arabic", "support_hi": "Support inbox, in Hindi",
          "delivery": "Late parcel", "injection": "Prompt injection", "agent": "AI agent guard",
          "review": "Mixed review"}
QUESTIONS = {  # question key -> (label on the site, option labels; None means the score legend's own words)
    "team": ("Which team handles it?", TEAMS),
    "refund": ("Asks for money back?", None),
    "urgency": ("How urgent?", None),
    "injection": ("Tries to override an AI's instructions?", None),
    "action": ("What should happen?", {"publish": "Publish", "hold": "Hold for review", "remove": "Remove",
                                       "approve": "Approve", "ask_human": "Ask a person", "block": "Block"}),
    "credentials": ("Asks for credentials?", None),
    "stars": ("Overall rating", None),
    "followup": ("What should the store do?", {"thank": "Thank them", "reach_out": "Reach out to fix it",
                                               "ignore": "No response needed"}),
    "defect": ("Reports a defect?", None),
    "praise": ("Praises anything?", None),
    "contacted": ("Tried to contact support?", None),
}
LABELS = {("injection", "action"): "What should moderation do?", ("agent", "action"): "Should the agent run it?"}


def state_text(state) -> str:
    if isinstance(state, dict):
        return " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in state.items())
    return str(state)


def question(example: str, key: str, answer: dict) -> dict:
    label, names = QUESTIONS[key]
    label = LABELS.get((example, key), label)
    if answer["type"] == "noul":
        opts = [("Yes", answer["noul"]), ("No", round(1 - answer["noul"], 4))]
    elif answer["type"] == "score":
        opts = [(answer["legend"][k], p) for k, p in answer["probabilities"].items()]
    else:
        opts = [(names[k] if names else k, p) for k, p in answer["probabilities"].items()]
    opts.sort(key=lambda o: -o[1])
    q = {"label": label, "kind": answer["type"], "options": [{"label": l, "probability": round(p, 4)} for l, p in opts]}
    if answer.get("reason"):
        q["reason"] = answer["reason"]
    return q


def ts(value, indent: int) -> str:
    """JSON rendered as the TypeScript style of unee.ts (unquoted keys, trailing commas)."""
    pad = " " * indent
    if isinstance(value, list):
        if not value:
            return "[]"
        inline = all(isinstance(v, dict) and set(v) == {"label", "probability"} for v in value)
        if inline:
            return "[\n" + "".join(f"{pad}  {{ label: {json.dumps(v['label'], ensure_ascii=False)}, probability: "
                                     f"{v['probability']} }},\n" for v in value) + pad + "]"
        return "[\n" + "".join(f"{pad}  {ts(v, indent + 2)},\n" for v in value) + pad + "]"
    if isinstance(value, dict):
        return "{\n" + "".join(f"{pad}  {k}: {ts(v, indent + 2)},\n" for k, v in value.items()) + pad + "}"
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--web", help="unee.ts to update in place (the reel array is replaced)")
    ap.add_argument("--examples", nargs="+", default=["support", "injection", "review"])
    ap.add_argument("--skip", nargs="*", default=[], help="<example>.<question> to leave off the site")
    args = ap.parse_args()
    rec = json.loads(Path(args.recording).read_text(encoding="utf-8"))
    for name, r in rec.items():  # every answer, so wrong ones are seen before choosing what to show
        print(name, {q: a.get("choice", a.get("noul", a.get("score"))) for q, a in r["response"]["answers"].items()})
    reel = []
    for name in args.examples:
        r = rec[name]
        qs = [question(name, k, a) for k, a in r["response"]["answers"].items()
              if f"{name}.{k}" not in args.skip and k in QUESTIONS]
        reel.append({"id": name, "title": TITLES[name], "input": state_text(r["request"]["state"]), "questions": qs})
    block = "  reel: " + ts(reel, 2) + " satisfies ReelExample[],"
    if not args.web:
        print(block)
        return
    path = Path(args.web)
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"  reel: \[\n.*?\n  \] satisfies ReelExample\[\],", lambda _: block, text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("reel block not found in " + args.web)
    path.write_text(new, encoding="utf-8")
    print(f"reel updated in {args.web}: {', '.join(args.examples)}")


if __name__ == "__main__":
    main()
