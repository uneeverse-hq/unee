---
title: Unee API
emoji: 🟣
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Free demo API for Unee, a small open decision and chat model
---

# Unee API (free demo)

Unee, from UNEEVERSE, by Muneef Mumthas. A small open AI model that makes decisions and streams chat answers from your own knowledge.

This Space runs **Unee 0.8B on Hugging Face's free CPU tier** (2 vCPU). It's meant for trying Unee: expect about 1–3 seconds per decision and a pause while the Space wakes up. For real traffic, run Unee yourself (`pip install unee`, then `unee serve`), or in the browser, where it costs nothing per call.

## Endpoints

- `POST /v1/systemone`: Jev-compatible decisions (`noul`, `choice`, `score`; many questions per call; `"explain": true` adds a reason)
- `POST /v1/chat/completions`: OpenAI-compatible chat, streamed with `"stream": true`. Add `"knowledge": [...]` to answer from your own documents.
- `GET /health`

```bash
curl -s https://<owner>-unee-api.hf.space/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for order #4821",
  "questions": {"team": {"type": "choice", "instructions": "Which team handles this?",
                         "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"}}}
}'
```

## Deploying this Space

1. Create a Docker Space (for example `uneeverse/unee-api`) on the free CPU Basic hardware.
2. Push this folder's `README.md` and `Dockerfile` to it. It builds from the published `unee` package and model files, so nothing else is needed.
