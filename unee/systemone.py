"""Jev `/v1/systemone` request parsing and response formatting (see docs/jev-api.md).

Every question becomes one decision task: options with descriptions, optional worked examples per option, and the
shared state. Scoring happens elsewhere; this module only translates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from unee.prompt import LETTERS

MAX_OPTIONS = 255  # like Jev; more than GROUP options are decided as a tournament (unee/server.py)
NOUL_DEFAULT = {"true": "Yes: the statement in the question holds for this input.",
                "false": "No: the statement in the question does not hold for this input."}


class RequestError(ValueError):
    """A 422: `field` names the offending part of the request."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


@dataclass
class Task:
    qid: str
    kind: str  # noul | choice | score
    question: str
    options: list[dict]  # {"key", "description"}
    examples: list[tuple] = field(default_factory=list)  # (state, key)
    explain: bool = False


def _text(value) -> str:
    """Instructions and descriptions may be structured JSON, as Jev allows. They are shown to the model as
    compact JSON, except {"system", "instruction"}, which becomes two paragraphs. Mirrors asText in systemone.js."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value) == {"system", "instruction"}:
        return f"{_text(value['system'])}\n\n{_text(value['instruction'])}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _criteria_entry(field_name: str, value) -> tuple[str, list]:
    """A criterion is a description (a string, or structured JSON), or {"what": description, "examples": [...]}."""
    if isinstance(value, str):
        return value, []
    if isinstance(value, dict) and isinstance(value.get("what"), str):
        ex = value.get("examples") or []
        if not isinstance(ex, list):
            raise RequestError(field_name, "examples must be a list")
        return value["what"], ex
    if isinstance(value, (dict, list)) and value:
        return _text(value), []
    raise RequestError(field_name, "must be a string or {\"what\": ..., \"examples\": [...]}")


def parse(body: dict) -> tuple[object, list[Task]]:
    if not isinstance(body, dict):
        raise RequestError("body", "must be a JSON object")
    if "state" not in body:
        raise RequestError("state", "is required")
    state = body["state"]
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions", "must be a non-empty object of question_id -> question")
    tasks = []
    for qid, q in questions.items():
        where = f"questions.{qid}"
        if not isinstance(q, dict):
            raise RequestError(where, "must be an object")
        kind = q.get("type")
        instructions = q.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or not _text(instructions).strip() or not instructions:
            raise RequestError(f"{where}.instructions", "is required")
        instructions = _text(instructions)
        criteria = q.get("criteria")
        examples: list[tuple] = []
        if kind == "noul":
            crit = criteria if criteria is not None else NOUL_DEFAULT
            if not isinstance(crit, dict) or set(crit) != {"true", "false"}:
                raise RequestError(f"{where}.criteria", 'noul criteria must have exactly the keys "true" and "false"')
            options = []
            for key in ("true", "false"):
                desc, ex = _criteria_entry(f"{where}.criteria.{key}", crit[key])
                options.append({"key": key, "description": desc})
                examples += [(s, key) for s in ex]
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise RequestError(f"{where}.criteria", "choice needs at least 2 options")
            options = []
            for key, value in criteria.items():
                desc, ex = _criteria_entry(f"{where}.criteria.{key}", value)
                options.append({"key": str(key), "description": desc})
                examples += [(s, str(key)) for s in ex]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise RequestError(f"{where}.criteria", "score needs a list of 2-10 levels, lowest first")
            options = []
            for i, value in enumerate(criteria):
                desc, ex = _criteria_entry(f"{where}.criteria.{i}", value)
                options.append({"key": str(i), "description": desc})
                examples += [(s, str(i)) for s in ex]
        else:
            raise RequestError(f"{where}.type", 'must be "noul", "choice" or "score"')
        if len(options) > MAX_OPTIONS:
            raise RequestError(f"{where}.criteria", f"at most {MAX_OPTIONS} options are supported")
        tasks.append(Task(str(qid), kind, instructions.strip(), options, examples, bool(q.get("explain"))))
    return state, tasks


def confidence(probs: list[float]) -> float:
    """1 - normalised entropy: 1 when one option has all the mass, 0 when it is spread evenly."""
    n = len(probs)
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return round(max(0.0, 1 - h / math.log(n)), 4) if n > 1 else 1.0


def answer(task: Task, probs: dict[str, float]) -> dict:
    p = [probs[o["key"]] for o in task.options]
    if task.kind == "noul":
        return {"type": "noul", "noul": round(probs["true"], 4)}
    if task.kind == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get),
                "probabilities": {k: round(v, 4) for k, v in probs.items()}, "confidence": confidence(p)}
    return {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4),
            "legend": {o["key"]: o["description"] for o in task.options},
            "probabilities": {k: round(v, 4) for k, v in probs.items()}, "confidence": confidence(p)}


def state_value(state) -> object:
    """Strings pass through; other JSON values are kept as JSON so the prompt shows their structure."""
    return state if isinstance(state, str) else json.loads(json.dumps(state))
