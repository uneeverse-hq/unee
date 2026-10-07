"""Triage support messages: many decisions about one message in a single call.

    unee serve --model unee-0.8b-Q4_K_M.gguf
    python examples/support_triage.py
"""

from unee import Client

unee = Client()
message = ("Hi, I ordered a standing desk on 2026-09-02 and it arrived on 2026-09-20 with a cracked leg. "
           "I want my money back, this is the second time. Order #88231.")
answers = unee.decide(message, {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments, refunds, invoices", "shipping": "Delivery delays, damaged or lost parcels",
                          "technical": "App or website problems", "sales": "Pre-purchase questions"}},
    "wants_refund": {"type": "noul", "instructions": "Does the customer ask for their money back?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait a week", "Within two days", "Today", "Immediately"]},
    "angry": {"type": "noul", "instructions": "Is the customer frustrated or angry?"},
})
for name, a in answers.items():
    value = {"choice": lambda: a["choice"], "noul": lambda: f"{a['noul']:.0%} yes",
             "score": lambda: f"level {a['score']:.1f}"}[a["type"]]()
    print(f"{name:>13}: {value}")
