"""Languages for multilingual training data, weighted roughly by how many people use them online.

Qwen3.5 (Unee's base) was trained on 201 languages and dialects; these are the ones the 9B teacher writes well enough
to teach from. The list leans towards widely used languages but deliberately includes lower-resource ones.
"""

from __future__ import annotations

import random

LANGUAGES = [
    # The most-spoken languages come first and carry most of the weight (owner's priority, 2026-10-05).
    ("Arabic", 12), ("Hindi", 12), ("Spanish", 12), ("Chinese (Simplified)", 12), ("French", 9),
    ("Portuguese (Brazil)", 9), ("Bengali", 8), ("Russian", 8), ("Urdu", 7), ("Indonesian", 7), ("German", 7),
    ("Japanese", 7),
    # Then a long tail, so Unee does not fall apart outside the top languages.
    ("Korean", 3), ("Turkish", 3),
    ("Italian", 3), ("Vietnamese", 3), ("Tamil", 2), ("Sinhala", 2), ("Thai", 2),
    ("Polish", 2), ("Dutch", 2), ("Ukrainian", 2), ("Persian", 2), ("Malay", 2), ("Filipino", 2), ("Swahili", 2),
    ("Hebrew", 1), ("Greek", 1), ("Czech", 1), ("Romanian", 1), ("Hungarian", 1), ("Swedish", 1), ("Telugu", 1),
    ("Marathi", 1), ("Gujarati", 1), ("Punjabi", 1), ("Nepali", 1), ("Chinese (Traditional)", 1), ("Amharic", 1),
    ("Yoruba", 1), ("Hausa", 1), ("Zulu", 1), ("Kazakh", 1), ("Burmese", 1), ("Khmer", 1),
]


def pick(rng: random.Random, share: float = 1.0) -> str | None:
    """A weighted random language, or None (English) with probability 1 - share."""
    if rng.random() >= share:
        return None
    names, weights = zip(*LANGUAGES)
    return rng.choices(names, weights=weights)[0]
