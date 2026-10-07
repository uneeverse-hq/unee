"""Rule-labelled date-window decisions: returns, warranties, trials, event refunds, leave notice, late fees,
rebooking, library fines, support SLAs and claim deadlines. Labels are computed, not guessed, so they are exact.

    PYTHONUTF8=1 .venv/Scripts/python data/date_rules.py --n 300 --out data/generated/dates.jsonl

Each template draws its own thresholds, names and date formats. It writes one solved example per option, plus
contrastive pairs that move a date across a threshold by a day or two (the edits single-pass models get wrong),
and sometimes adds an unrelated distractor date. Output rows have the same fields as gen.py's train.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
NAMES = ["Ana Ruiz", "Ben Okafor", "Chloe Martin", "Dev Patel", "Elif Kaya", "Femi Adeyemi", "Grace Lin", "Hugo Brandt",
         "Isla Moreau", "Jonas Berg", "Kemi Bello", "Liam Walsh", "Mei Tanaka", "Nadia Haddad", "Omar Farouk",
         "Priya Nair", "Quinn Reyes", "Rosa Delgado", "Sam Whitfield", "Tara Singh", "Umar Sheikh", "Vera Novak"]
SHOPS = ["Pinewood Outfitters", "Brightline Electronics", "Calder Home", "Northfield Books", "Lumen Audio",
         "Harbor Kitchenware", "Summit Cycles", "Velvet Pine Apparel", "Orbit Toys", "Granite Tools"]
ITEMS = ["a rain jacket", "a pair of headphones", "a desk lamp", "a coffee grinder", "a tent", "a blender",
         "a backpack", "a monitor", "a cast-iron pan", "a smartwatch", "a winter coat", "a mechanical keyboard"]


def fmt(d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "mdy":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "dmy":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return f"{MONTHS[d.month - 1]} {d.day}"  # year-less, used only after a dated mention


def plural(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


# Each scenario factory returns: draw(rng) -> params; options(p) -> [(key, description)]; gold(p, days) -> key;
# text(p, rng, d0, d1, f0, f1) -> state sentence(s); why(p, days, key) -> reason. `days` = d1 - d0.
def _returns():
    def draw(rng):
        a = rng.choice([7, 10, 14, 15, 20, 30])
        b = a + rng.choice([15, 16, 20, 30, 45])
        return {"a": a, "b": b, "shop": rng.choice(SHOPS), "item": rng.choice(ITEMS)}
    return {
        "draw": draw,
        "family": "rule_returns_window",
        "question": lambda p: "Under this return policy, what should the decision on the request be?",
        "policy": lambda p: (f"Policy ({p['shop']}): returns requested no more than {p['a']} days after delivery get a "
                             f"full refund; returns requested {p['a'] + 1} to {p['b']} days after delivery get store "
                             f"credit only; no returns are accepted more than {p['b']} days after delivery."),
        "options": lambda p: [("approve_full_refund", f"The return was requested at most {p['a']} days after delivery."),
                              ("approve_store_credit", f"The return was requested {p['a'] + 1} to {p['b']} days after delivery."),
                              ("deny_outside_window", f"The return was requested more than {p['b']} days after delivery.")],
        "gold": lambda p, n: "approve_full_refund" if n <= p["a"] else "approve_store_credit" if n <= p["b"] else "deny_outside_window",
        "bounds": lambda p: [p["a"], p["b"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"{p['item'].capitalize()} was delivered on {f0}. The customer asked to return it on {f1}. It is unused.",
            f"Customer {rng.choice(NAMES)} received {p['item']} on {f0} and requested a return on {f1}.",
            f"Return request logged {f1} for {p['item']} delivered {f0}; item in original packaging."]),
        "why": lambda p, n, k: f"The request came {plural(n)} after delivery, so the {k.replace('_', ' ')} rule applies.",
    }


def _warranty():
    def draw(rng):
        return {"n": rng.choice([30, 60, 90, 180, 365]), "item": rng.choice(ITEMS), "shop": rng.choice(SHOPS)}
    return {
        "draw": draw,
        "family": "rule_warranty",
        "question": lambda p: f"Is this repair claim covered by the {p['n']}-day warranty, which counts from the purchase date?",
        "policy": lambda p: f"Warranty ({p['shop']}): defects reported within {p['n']} days of purchase are repaired free.",
        "options": lambda p: [("covered", f"The defect was reported no more than {p['n']} days after purchase."),
                              ("not_covered", f"The defect was reported more than {p['n']} days after purchase.")],
        "gold": lambda p, n: "covered" if n <= p["n"] else "not_covered",
        "bounds": lambda p: [p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Bought {p['item']} on {f0}. It stopped working and the owner reported it on {f1}.",
            f"Purchase date: {f0}. Fault reported: {f1}. Product: {p['item']}.",
            f"{rng.choice(NAMES)} filed a repair ticket on {f1} for {p['item']} purchased {f0}."]),
        "why": lambda p, n, k: f"The fault was reported {plural(n)} after purchase, against a {p['n']}-day warranty.",
    }


def _trial():
    def draw(rng):
        return {"n": rng.choice([7, 14, 21, 30]), "app": rng.choice(["StreamBox", "FitPulse", "LedgerLite", "Notely", "TuneHub"])}
    return {
        "draw": draw,
        "family": "rule_trial_cancel",
        "question": lambda p: f"Should this {p['app']} cancellation be charged?",
        "policy": lambda p: f"Free trial: cancelling within {p['n']} days of sign-up costs nothing; after that the first month is billed.",
        "options": lambda p: [("no_charge", f"Cancelled no more than {p['n']} days after sign-up."),
                              ("charge_first_month", f"Cancelled more than {p['n']} days after sign-up.")],
        "gold": lambda p, n: "no_charge" if n <= p["n"] else "charge_first_month",
        "bounds": lambda p: [p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"User signed up on {f0} and cancelled on {f1}.",
            f"Account created {f0}; cancellation received {f1}; plan: monthly.",
            f"{rng.choice(NAMES)} started the trial {f0} and asked to cancel on {f1}."]),
        "why": lambda p, n, k: f"The cancellation came {plural(n)} after sign-up; the free window is {p['n']} days.",
    }


def _event():
    def draw(rng):
        b = rng.choice([2, 3, 7])
        a = b + rng.choice([7, 11, 14, 21])
        return {"a": a, "b": b, "event": rng.choice(["the concert", "the conference", "the cooking class", "the match", "the workshop"])}
    return {
        "draw": draw,
        "family": "rule_event_refund",
        "question": lambda p: f"What refund does this cancelled booking for {p['event']} get?",
        "policy": lambda p: (f"Refunds: cancel at least {p['a']} days before the event for a full refund, {p['b']} to "
                             f"{p['a'] - 1} days before for a 50% refund, and fewer than {p['b']} days before for no refund."),
        "options": lambda p: [("full_refund", f"Cancelled {p['a']} or more days before the event."),
                              ("half_refund", f"Cancelled {p['b']} to {p['a'] - 1} days before the event."),
                              ("no_refund", f"Cancelled fewer than {p['b']} days before the event.")],
        # days = event - cancellation, so d0 is the cancellation and d1 the event
        "gold": lambda p, n: "full_refund" if n >= p["a"] else "half_refund" if n >= p["b"] else "no_refund",
        "bounds": lambda p: [p["b"] - 1, p["a"] - 1],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Booking for {p['event']} on {f1} was cancelled by the customer on {f0}.",
            f"Cancellation received {f0}. Event date: {f1}.",
            f"{rng.choice(NAMES)} cancelled on {f0}; {p['event']} takes place on {f1}."]),
        "why": lambda p, n, k: f"The booking was cancelled {plural(n)} before the event.",
    }


def _leave():
    def draw(rng):
        return {"n": rng.choice([7, 10, 14, 21, 30])}
    return {
        "draw": draw,
        "family": "rule_leave_notice",
        "question": lambda p: "How should this leave request be handled?",
        "policy": lambda p: (f"Leave policy: requests made at least {p['n']} days before the first day of leave are approved "
                             f"automatically; requests with less notice go to the manager."),
        "options": lambda p: [("auto_approve", f"Requested {p['n']} or more days before the leave starts."),
                              ("manager_review", f"Requested fewer than {p['n']} days before the leave starts.")],
        "gold": lambda p, n: "auto_approve" if n >= p["n"] else "manager_review",
        "bounds": lambda p: [p["n"] - 1],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"{rng.choice(NAMES)} submitted a leave request on {f0} for leave starting {f1}.",
            f"Request date: {f0}. First day of leave: {f1}. Type: annual leave.",
            f"Leave starting {f1} was requested on {f0}."]),
        "why": lambda p, n, k: f"The request gave {plural(n)} of notice against a {p['n']}-day requirement.",
    }


def _late_fee():
    def draw(rng):
        return {"n": rng.choice([5, 7, 10, 15, 30])}
    return {
        "draw": draw,
        "family": "rule_late_fee",
        "question": lambda p: "What should happen to this invoice payment?",
        "policy": lambda p: (f"Billing: payments on or before the due date have no fee; payments 1 to {p['n']} days late "
                             f"get a late fee; payments more than {p['n']} days late trigger an account suspension."),
        "options": lambda p: [("no_fee", "Paid on or before the due date."),
                              ("late_fee", f"Paid 1 to {p['n']} days after the due date."),
                              ("suspend_account", f"Paid more than {p['n']} days after the due date.")],
        "gold": lambda p, n: "no_fee" if n <= 0 else "late_fee" if n <= p["n"] else "suspend_account",
        "bounds": lambda p: [0, p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Invoice due {f0}; payment received {f1}.",
            f"Due date: {f0}. Paid: {f1}. Amount: ${rng.randint(40, 900)}.",
            f"{rng.choice(SHOPS)} invoice #{rng.randint(1000, 9999)} was due on {f0} and paid on {f1}."]),
        "why": lambda p, n, k: ("The payment arrived on or before the due date." if n <= 0
                                else f"The payment arrived {plural(n)} after the due date."),
    }


def _rebook():
    def draw(rng):
        return {"n": rng.choice([3, 7, 14, 21])}
    return {
        "draw": draw,
        "family": "rule_rebooking",
        "question": lambda p: "What does this flight change cost under the fare rules?",
        "policy": lambda p: f"Fare rules: changes made at least {p['n']} days before departure are free; later changes cost a fee.",
        "options": lambda p: [("free_change", f"Changed {p['n']} or more days before departure."),
                              ("change_fee", f"Changed fewer than {p['n']} days before departure.")],
        "gold": lambda p, n: "free_change" if n >= p["n"] else "change_fee",
        "bounds": lambda p: [p["n"] - 1],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Passenger {rng.choice(NAMES)} asked on {f0} to change a flight departing {f1}.",
            f"Change request: {f0}. Original departure: {f1}.",
            f"Flight on {f1}; the change was requested {f0}."]),
        "why": lambda p, n, k: f"The change was requested {plural(n)} before departure.",
    }


def _library():
    def draw(rng):
        return {"n": rng.choice([3, 5, 7, 14])}
    return {
        "draw": draw,
        "family": "rule_library_fine",
        "question": lambda p: "What applies to this returned library book?",
        "policy": lambda p: (f"Library: books returned by the due date have no fine; 1 to {p['n']} days late pay a small fine; "
                             f"more than {p['n']} days late pause borrowing privileges."),
        "options": lambda p: [("no_fine", "Returned on or before the due date."),
                              ("small_fine", f"Returned 1 to {p['n']} days late."),
                              ("pause_borrowing", f"Returned more than {p['n']} days late.")],
        "gold": lambda p, n: "no_fine" if n <= 0 else "small_fine" if n <= p["n"] else "pause_borrowing",
        "bounds": lambda p: [0, p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Book due back {f0}, returned {f1}.",
            f"{rng.choice(NAMES)} returned 'The Long Harbor' on {f1}; the due date was {f0}.",
            f"Due: {f0}. Returned: {f1}. Condition: good."]),
        "why": lambda p, n, k: ("The book came back on time." if n <= 0 else f"The book came back {plural(n)} late."),
    }


def _sla():
    def draw(rng):
        return {"n": rng.choice([1, 2, 3, 5])}
    return {
        "draw": draw,
        "family": "rule_sla",
        "question": lambda p: f"Did this ticket meet the {p['n']}-day first-response target?",
        "policy": lambda p: f"Support SLA: the first response must be sent within {p['n']} days of the ticket being opened.",
        "options": lambda p: [("met", f"First response sent no more than {p['n']} days after opening."),
                              ("breached", f"First response sent more than {p['n']} days after opening.")],
        "gold": lambda p, n: "met" if n <= p["n"] else "breached",
        "bounds": lambda p: [p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Ticket opened {f0}; first agent reply {f1}.",
            f"Customer wrote in on {f0}. Our first response went out on {f1}.",
            f"Opened: {f0}. First response: {f1}. Priority: normal."]),
        "why": lambda p, n, k: f"The first reply came {plural(n)} after the ticket was opened.",
    }


def _claim():
    def draw(rng):
        return {"n": rng.choice([14, 30, 45, 60, 90])}
    return {
        "draw": draw,
        "family": "rule_claim_deadline",
        "question": lambda p: "Is this insurance claim within the filing deadline?",
        "policy": lambda p: f"Claims must be filed within {p['n']} days of the incident; later claims are rejected.",
        "options": lambda p: [("within_deadline", f"Filed no more than {p['n']} days after the incident."),
                              ("late_reject", f"Filed more than {p['n']} days after the incident.")],
        "gold": lambda p, n: "within_deadline" if n <= p["n"] else "late_reject",
        "bounds": lambda p: [p["n"]],
        "text": lambda p, rng, f0, f1: rng.choice([
            f"Incident on {f0} (water damage); claim filed {f1}.",
            f"{rng.choice(NAMES)} reported a car accident that happened {f0}. The claim was submitted {f1}.",
            f"Date of loss: {f0}. Claim received: {f1}."]),
        "why": lambda p, n, k: f"The claim was filed {plural(n)} after the incident.",
    }


FACTORIES = [_returns, _warranty, _trial, _event, _leave, _late_fee, _rebook, _library, _sla, _claim]


# Non-date conditions that override the date rule, so the model learns to read the whole policy, not just count days.
# Each: (policy clause, option key, option description, state sentence that triggers it, reason).
EXTRAS = {
    "rule_returns_window": [
        ("Items sold as final sale cannot be returned.", "deny_final_sale", "The item was sold as final sale.",
         "It was bought on clearance as a final-sale item.", "The item was a final-sale purchase, which cannot be returned."),
        ("Items that arrived damaged are always sent to a human agent.", "escalate_to_human",
         "The item arrived damaged.", "The customer says it arrived with a cracked casing.",
         "The item arrived damaged, which always goes to a human agent."),
    ],
    "rule_warranty": [
        ("Damage from drops or liquid is never covered.", "not_covered_damage", "The fault came from a drop or liquid.",
         "The owner admits it was dropped in water.", "Liquid damage is excluded whatever the date."),
    ],
    "rule_leave_notice": [
        ("Requests made on or after the first day of leave are denied.", "deny", "Requested on or after the first day of leave.",
         None, None),  # triggered by dates (n <= 0), see gold_with_extras
    ],
}
NEUTRAL = {
    "rule_returns_window": ["It is unused.", "Tags are still attached.", "It is in its original packaging.", ""],
    "rule_warranty": ["The screen flickers on start-up.", "It no longer charges.", ""],
}


def make_state(sc, p, rng, days: int, distractor: bool, policy: str, suffix: str = "") -> str:
    d0 = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    d1 = d0 + timedelta(days=days)
    style = rng.choice(["iso", "iso", "mdy", "dmy"])
    s1 = "md" if style != "iso" and d1.year == d0.year and rng.random() < 0.25 else style
    text = sc["text"](p, rng, fmt(d0, style), fmt(d1, s1))
    if suffix:
        text += " " + suffix
    if distractor:
        dx = d0 - timedelta(days=rng.randint(1, 40))
        text += rng.choice([f" The order was placed on {fmt(dx, style)}.", f" Account opened {fmt(dx, style)}.",
                            f" Previous contact: {fmt(dx, style)}."])
    if p["policy_in_state"]:
        text = policy + " " + text
    return text


def template(i: int, seed: int, p_extra: float = 0.5) -> list[dict]:
    rng = random.Random(f"dates-{seed}-{i}")
    sc = FACTORIES[i % len(FACTORIES)]()
    fam = sc["family"]
    p = sc["draw"](rng)
    p["policy_in_state"] = rng.random() < 0.5
    extras = [e for e in EXTRAS.get(fam, []) if rng.random() < 0.8]
    policy = " ".join([sc["policy"](p)] + [e[0] for e in extras])
    question = sc["question"](p) if p["policy_in_state"] else sc["question"](p) + " " + policy
    options = [{"key": k, "description": d} for k, d in sc["options"](p)] + [{"key": e[1], "description": e[2]} for e in extras]
    bounds = sc["bounds"](p)
    deny_late = any(e[1] == "deny" for e in extras)  # leave: on/after the first day

    def gold(n: int, extra=None) -> str:
        if extra is not None:
            return extra[1]
        if deny_late and n <= 0:
            return "deny"
        return sc["gold"](p, n)

    def neutral() -> str:
        return rng.choice(NEUTRAL.get(fam, [""])) if rng.random() < 0.6 else ""

    def days_for(key: str) -> int:
        for _ in range(500):
            n = rng.randint(-5, max(bounds) + 40)
            if gold(n) == key and (key != "no_fee" or n <= 0):
                return n
        raise ValueError(key)

    examples = []
    for o in options:
        ex = next((e for e in extras if e[1] == o["key"] and e[3]), None)
        n = days_for(o["key"]) if ex is None else days_for(sc["options"](p)[0][0])
        examples.append({"state": make_state(sc, p, rng, n, rng.random() < 0.3, policy, ex[3] if ex else neutral()),
                         "gold": o["key"]})

    rows = []

    def add(j, half, n, extra, edit, distract):
        g = gold(n, extra)
        why = extra[4] if extra is not None else sc["why"](p, n, g)
        rows.append({"tid": f"d{seed}-{i:05d}", "family": fam, "kind": "choice", "domain": "rules",
                     "question": question, "options": options, "examples": examples,
                     "id": f"d{seed}-{i:05d}-{j}{half}",
                     "state": make_state(sc, p, rng, n, distract, policy, extra[3] if extra is not None else neutral()),
                     "gold": g, "why": why, "teacher": {}, "edit": edit, "designed_gold": g})

    triggers = [e for e in extras if e[3]]
    for j in range(rng.choice([2, 3])):
        distract = rng.random() < 0.35
        if triggers and rng.random() < p_extra:  # condition pair: same dates, the condition decides
            n = days_for(sc["options"](p)[0][0])
            e = rng.choice(triggers)
            add(j, "a", n, e, "a policy condition applies", distract)
            add(j, "b", n, None, "a policy condition applies", distract)
        else:  # date pair: a date moves across a threshold
            b = rng.choice(bounds)
            n_a, n_b = b - rng.randint(0, 2), b + 1 + rng.randint(0, 2)
            if rng.random() < 0.5:
                n_a, n_b = n_b, n_a
            add(j, "a", n_a, None, "date moved across a threshold", distract)
            add(j, "b", n_b, None, "date moved across a threshold", distract)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="templates (each gives 4-8 items)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "generated" / "dates.jsonl"))
    args = ap.parse_args()
    rows = [r for i in range(args.n) for r in template(i, args.seed)]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from collections import Counter
    print(f"{len(rows)} items from {args.n} templates -> {args.out}")
    print(Counter(r["gold"] for r in rows).most_common(12))


if __name__ == "__main__":
    main()
