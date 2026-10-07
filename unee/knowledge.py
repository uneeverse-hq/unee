"""Built-in retrieval for knowledge-grounded chat: split documents into passages and rank them with BM25.

There's no index and no extra dependency. For the few hundred passages that an app's FAQ, product and policy pages
make, ranking them on every question takes milliseconds. `ground` turns chat messages plus documents into the
messages Unee was trained on (unee.prompt.knowledge_messages).

Words are runs of letters, digits and combining marks in any script, so Arabic, Hindi, Russian and the rest
retrieve like English. Scripts written without spaces (Chinese, Japanese, Thai, Lao, Khmer, Myanmar) are matched
on character pairs. Mirrored exactly in packages/js/src/knowledge.js.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from unee.prompt import knowledge_messages

_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or our so that the "
    "this to was we what when where which who will with you your".split())
# Kana, CJK, Thai and Lao, Myanmar, Khmer: no spaces between words.
_NO_SPACE = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x0E00, 0x0EFF),
             (0x1000, 0x109F), (0x1780, 0x17FF))
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


def _no_space(text: str) -> bool:
    return any(lo <= ord(c) <= hi for c in text for lo, hi in _NO_SPACE)


def _runs(text: str) -> list[str]:
    """Runs of letters, digits and combining marks (Unicode categories L, N, M), lower-cased."""
    out, cur = [], []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in "LNM":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _words(text: str) -> list[str]:
    out = []
    for w in _runs(text):
        if _no_space(w):
            out += [w[i:i + 2] for i in range(max(1, len(w) - 1))]
            continue
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):  # refunds -> refund
            w = w[:-1]
        out.append(w)
    return out


def _count(text: str) -> int:
    """Rough word count; text without spaces counts one word per two characters."""
    n = len(text.split())
    return max(n, len(text) // 2) if _no_space(text) else n


def passages(docs, max_words: int = 120) -> list[str]:
    """Split documents (strings, or {"title", "text"} dicts) into passages of up to about `max_words` words.
    Splits happen at paragraph breaks, or at sentence breaks inside long paragraphs. Each passage starts with
    its document's title."""
    out = []
    for d in docs:
        title, text = (d.get("title") or "", d.get("text") or "") if isinstance(d, dict) else ("", str(d))
        units = []
        for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
            if p:
                units += [u for u in _SENTENCE_END.split(p) if u] if _count(p) > max_words else [p]
        cur: list[str] = []
        n = 0
        for u in units:
            k = _count(u)
            if cur and n + k > max_words:
                out.append((f"{title}: " if title else "") + "\n".join(cur))
                cur, n = [], 0
            cur.append(u)
            n += k
        if cur:
            out.append((f"{title}: " if title else "") + "\n".join(cur))
    return out


def search(query: str, chunks: list[str], k: int = 4) -> list[str]:
    """The `k` passages that best match `query` (BM25), best first. Passages with no word in common are dropped."""
    if len(chunks) <= k:
        return list(chunks)
    docs = [_words(c) for c in chunks]
    avg = sum(map(len, docs)) / len(docs) or 1.0
    df = Counter(w for d in docs for w in set(d))
    q = list(dict.fromkeys(_words(query)))  # first-occurrence order, so sums match the JS port exactly
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2
                / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg)) for w in q if tf[w])
        if s > 0:
            scored.append((-s, i))
    return [chunks[i] for _, i in sorted(scored)[:k]]


def ground(messages: list[dict], docs, k: int = 4) -> list[dict]:
    """Chat messages with the passages most relevant to the conversation added as knowledge. The query is the
    last two user turns, so a short follow-up ("and how much is it?") still finds its topic."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    return knowledge_messages(messages, search(" ".join(users[-2:]), passages(docs), k))
