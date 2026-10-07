"""Read an Open Knowledge Format (OKF) bundle as knowledge for Unee.

    from unee import Client
    from unee.okf import load_bundle

    docs = load_bundle("help-centre/")          # a folder of Markdown files with YAML front matter
    for piece in Client().chat("How long do refunds take?", knowledge=docs, strict=True): print(piece, end="")

OKF (github.com/GoogleCloudPlatform/open-knowledge-format, v0.2 when this was written) is a folder of UTF-8 Markdown
files. Each concept file starts with a front matter block between `---` lines whose only required key is `type`;
`title`, `description`, `tags` and `status` are among the optional ones. `index.md` and `log.md` are reserved (a
directory listing and a change log) and are not concepts.

Each concept becomes one Unee document: its title (with the folder it sits in) and its description plus body. The
reader is permissive, as the spec asks: unknown keys and types, missing front matter and broken links never reject
a bundle. Concepts marked `status: deprecated` are left out unless asked for. There is no YAML dependency: only the
top-level scalar and list keys are read, which is all this needs. Mirrored in packages/js/src/okf.js.
"""

from __future__ import annotations

import re
from pathlib import Path

RESERVED = {"index.md", "log.md"}
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):[ \t]*(.*)$")


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def front_matter(text: str) -> tuple[dict, str]:
    """The top-level front matter keys of a concept (strings, or lists of strings) and its Markdown body. Nested
    mappings are skipped. Text with no front matter block comes back whole, with no keys."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    meta: dict = {}
    key, block = None, []

    def nested(item: str) -> bool:  # "- resource: ..." starts a mapping, not a string
        return ":" in item[2:] and item[2:].strip()[:1] not in "\"'"

    def close() -> None:
        if key is None or key in meta:
            return
        items = [b.strip() for b in block if b.strip()]
        if items and all(i.startswith("- ") for i in items) and not any(map(nested, items)):
            meta[key] = [_scalar(i[2:]) for i in items]
        elif items and not any(_KEY.match(i) or i.startswith("- ") for i in items):
            meta[key] = " ".join(items)  # a folded or literal block scalar

    for line in lines[1:end]:
        m = _KEY.match(line)
        if m and not line[:1].isspace():
            close()
            key, block = m.group(1), []
            value = m.group(2).strip()
            if value in (">", "|", ">-", "|-", ">+", "|+", ""):
                continue
            if value.startswith("[") and value.endswith("]"):
                meta[key] = [_scalar(v) for v in value[1:-1].split(",") if v.strip()]
            elif not value.startswith("{"):
                meta[key] = _scalar(value)
        elif key is not None:
            block.append(line)
    close()
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def documents(files: dict[str, str], include_deprecated: bool = False) -> list[dict]:
    """Unee documents ({"title", "text"}) for the concept files of a bundle, given as {path in the bundle: file
    text}. Reserved files and non-Markdown files are ignored."""
    out = []
    for path in sorted(files):
        parts = path.replace("\\", "/").strip("/").split("/")
        name = parts[-1]
        if not name.lower().endswith(".md") or name.lower() in RESERVED:
            continue
        meta, body = front_matter(files[path])
        if str(meta.get("status", "")).lower() == "deprecated" and not include_deprecated:
            continue
        title = meta.get("title") if isinstance(meta.get("title"), str) and meta.get("title") else name[:-3].replace("_", " ").replace("-", " ")
        where = " / ".join(parts[:-1])
        head = [str(meta["description"])] if isinstance(meta.get("description"), str) and meta.get("description") else []
        if isinstance(meta.get("tags"), list) and meta["tags"]:
            head.append("Tags: " + ", ".join(meta["tags"]))
        text = "\n\n".join(head + [body.strip()]).strip()
        if text:
            out.append({"title": f"{where} / {title}" if where else title, "text": text})
    return out


def load_bundle(root: str | Path, include_deprecated: bool = False) -> list[dict]:
    """Unee documents for the OKF bundle in the folder `root`."""
    root = Path(root)
    files = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
             for p in root.rglob("*.md") if p.is_file()}
    return documents(files, include_deprecated)
