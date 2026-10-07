# Jev `/v1/systemone` request and response shape (reference for our compatible server)

Sources:
- the write-up at https://flaviocopes.com/jev/ (read on 2026-10-04)
- DecideBench's adapter, `bench/decidebench/decidebench/systems/jev.py`

TypeSafe's own docs weren't checked directly. Re-verify this against them before calling the server "compatible".

## Request

`POST /v1/systemone`, `Authorization: Bearer <key>`

```json
{
  "model": "jev-latest",
  "state": "<string> | {object} | [array]",
  "questions": {
    "<question_id>": { "type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."} },
    "<question_id>": { "type": "choice", "instructions": "...", "criteria": {"<key>": "<description>"} },
    "<question_id>": { "type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", "..."] }
  }
}
```

Notes on the fields:
- **Noul criteria** are optional.
- **Choice examples:** a choice criterion can be `{"what": "<description>", "examples": ["<state>", ...]}` instead of a plain string. DecideBench passes worked examples this way.
- **Limits:**
  - state plus the longest question must fit in about 32k tokens
  - up to 255 choice options
  - 2–10 score levels
  - text only.

## Response

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "<noul_id>":   { "type": "noul", "noul": 0.99 },
    "<choice_id>": { "type": "choice", "choice": "<key>", "probabilities": {"<key>": 0.97}, "confidence": 0.95 },
    "<score_id>":  { "type": "score", "score": 1.9, "legend": {"0": "...", "1": "...", "2": "..."},
                     "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}, "confidence": 0.86 }
  },
  "usage": { "input_tokens": 210, "output_tokens": 31 }
}
```

How the fields are defined:
- `noul` is P(yes).
- `choice` is the arg-max of `probabilities`.
- `score` is the probability-weighted mean level, so it can fall between levels.
- `confidence` is high when one option dominates.

## Errors

| Code | Meaning |
|---|---|
| 401 | Bad or missing key |
| 422 | Validation failure; the response names the field |
| 429 | Rate limited |
| 529 | Overloaded |
