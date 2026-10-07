"""Deterministic facts computed from the input, added to the prompt so a small model compares numbers instead of
doing calendar arithmetic in one forward pass (single-pass models, Jev included, are weak at dates).

    >>> date_facts("Delivered on 2026-05-01. Return requested on 2026-06-05.")
    ['2026-06-05 is 35 days after 2026-05-01']

Recognised: ISO dates (2026-05-01), "May 1, 2026" / "May 1st 2026", "1 May 2026", and "May 1" without a year when
another date in the text gives the year. Ambiguous numeric forms such as 05/01/2026 are skipped on purpose.
Mirrored in packages/js/src/facts.js; both must produce identical strings.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
PATTERNS = [
    ("iso", re.compile(r"\b((?:19|20)\d\d)-(\d\d)-(\d\d)\b")),
    ("mdy", re.compile(r"\b" + _MON + r" " + _DAY + r",? ((?:19|20)\d\d)\b")),
    ("dmy", re.compile(r"\b" + _DAY + r" " + _MON + r",? ((?:19|20)\d\d)\b")),
    ("md", re.compile(r"\b" + _MON + r" " + _DAY + r"\b(?!,? (?:19|20)\d\d)(?!:)")),
]
MAX_FACTS = 6

# Other languages: day-month(-year) with local month names, and the CJK year-month-day forms. Names that are also
# English month names or abbreviations stay with the English patterns above, so English text is never affected.
# Digits may be Western, Arabic-Indic, Persian, Devanagari or Bengali. Mirrored in packages/js/src/facts.js.
ML_MONTHS = {
    # Spanish, Portuguese, Italian
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
    "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12, "janeiro": 1, "fevereiro": 2, "março": 3,
    "maio": 5, "junho": 6, "julho": 7, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12, "gennaio": 1,
    "febbraio": 2, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7, "settembre": 9, "ottobre": 10, "dicembre": 12,
    # French, German, Dutch
    "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12, "januari": 1, "februari": 2, "maart": 3, "mei": 5, "augustus": 8,
    # Indonesian, Malay, Turkish
    "maret": 3, "agustus": 8, "desember": 12, "julai": 7, "ogos": 8, "disember": 12, "ocak": 1, "şubat": 2,
    "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6, "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11,
    "aralık": 12,
    # Russian and Ukrainian (as written after a day number)
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12, "січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5,
    "червня": 6, "липня": 7, "серпня": 8, "вересня": 9, "жовтня": 10, "листопада": 11, "грудня": 12,
    # Arabic, Urdu
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7, "أغسطس": 8,
    "اغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12, "جنوری": 1, "فروری": 2,
    "مارچ": 3, "اپریل": 4, "مئی": 5, "جون": 6, "جولائی": 7, "اگست": 8, "ستمبر": 9, "اکتوبر": 10, "نومبر": 11,
    "دسمبر": 12,
    # Hindi, Bengali
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
    "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
    "জানুয়ারি": 1, "ফেব্রুয়ারি": 2, "মার্চ": 3, "এপ্রিল": 4, "মে": 5, "জুন": 6, "জুলাই": 7, "আগস্ট": 8,
    "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12,
}
_DIG = "[0-9٠-٩۰-۹०-९০-৯]"
_ML_MON = "(" + "|".join(sorted(ML_MONTHS, key=lambda s: (-len(s), s))) + ")"
_ML_DAY = rf"(?<!\w)({_DIG}{{1,2}})(?:er)?\.?\s(?:de\s)?"
_ML_YEAR = rf"(?:\s(?:de\s|del\s)?|,\s?)({_DIG}{{4}})(?!\w)"
ML_PATTERNS = [
    ("dmy", re.compile(_ML_DAY + _ML_MON + r"\.?" + _ML_YEAR, re.IGNORECASE)),
    ("ymd", re.compile(rf"({_DIG}{{4}})\s?[年년]\s?({_DIG}{{1,2}})\s?[月월]\s?({_DIG}{{1,2}})\s?[日일]")),
    ("dm", re.compile(_ML_DAY + _ML_MON + r"(?!\w)(?!\.?" + _ML_YEAR + ")", re.IGNORECASE)),
    ("md", re.compile(rf"(?<!\w)({_DIG}{{1,2}})\s?[月월]\s?({_DIG}{{1,2}})\s?[日일]")),
]


def _num(s: str) -> int:
    return int(s)  # int() reads every Unicode decimal digit; the JS port maps the same scripts by hand


def _ml_match(kind: str, m) -> tuple[date | None, tuple[int, int] | None]:
    if kind == "dmy":
        year = _num(m[3])
        if not 1900 <= year <= 2099:
            raise ValueError("year out of range")
        return date(year, ML_MONTHS[m[2].lower()], _num(m[1])), None
    if kind == "ymd":
        year = _num(m[1])
        if not 1900 <= year <= 2099:
            raise ValueError("year out of range")
        return date(year, _num(m[2]), _num(m[3])), None
    md = (ML_MONTHS[m[2].lower()], _num(m[1])) if kind == "dm" else (_num(m[1]), _num(m[2]))
    date(2000, *md)  # validates month/day as for English year-less dates
    return None, md


def find_dates(text: str) -> list[tuple[int, str, date | None, tuple[int, int] | None]]:
    """(position, matched text, date or None, (month, day) for year-less dates), in order, no overlaps."""
    found, taken = [], []
    for kind, pat in [*PATTERNS, *[("ml-" + k, p) for k, p in ML_PATTERNS]]:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            try:
                if kind.startswith("ml-"):
                    d, md = _ml_match(kind[3:], m)
                elif kind == "iso":
                    d, md = date(int(m[1]), int(m[2]), int(m[3])), None
                elif kind == "mdy":
                    d, md = date(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2])), None
                elif kind == "dmy":
                    d, md = date(int(m[3]), MONTHS[m[2][:3].lower()], int(m[1])), None
                else:
                    md = (MONTHS[m[1][:3].lower()], int(m[2]))
                    date(2000, *md)  # validates month/day (2000 is a leap year, so Feb 29 passes)
                    d = None
            except (ValueError, KeyError):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.group(0), d, md))
    return sorted(found)


def date_facts(text: str) -> list[str]:
    """Day differences between the distinct dates in `text`, earliest-mentioned first, at most MAX_FACTS."""
    if not isinstance(text, str):
        return []
    raw = find_dates(text)
    years = [d.year for _, _, d, _ in raw if d]
    resolved: list[tuple[str, date]] = []
    for _, s, d, md in raw:
        if d is None:
            if not years:
                continue  # a lone "May 3" has no year to anchor it
            try:
                d = date(years[0], *md)
            except ValueError:
                continue
        if all(s != t for t, _ in resolved):
            resolved.append((s, d))
    facts = []
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            (si, di), (sj, dj) = resolved[i], resolved[j]
            n = (dj - di).days
            if n == 0:
                facts.append(f"{sj} is the same day as {si}")
            else:
                word = "day" if abs(n) == 1 else "days"
                facts.append(f"{sj} is {abs(n)} {word} {'after' if n > 0 else 'before'} {si}")
            if len(facts) == MAX_FACTS:
                return facts
    return facts
