// Read an Open Knowledge Format (OKF) bundle as knowledge for Unee. A port of unee/okf.py that reads identically.
//
//   import { okfDocuments } from "unee";
//   const docs = okfDocuments({ "refunds.md": text1, "shipping/uk.md": text2 });   // path in the bundle -> file text
//   for await (const piece of unee.stream(messages, { knowledge: docs, strict: true })) ...
//
// OKF (github.com/GoogleCloudPlatform/open-knowledge-format, v0.2 when this was written) is a folder of Markdown
// files. Each concept starts with a front matter block between `---` lines whose only required key is `type`.
// `index.md` and `log.md` are reserved and are not concepts. The reader is permissive, as the spec asks: unknown
// keys and types, missing front matter and broken links never reject a bundle. Concepts marked
// `status: deprecated` are left out unless asked for. Only top-level scalar and list keys are read.

const RESERVED = new Set(["index.md", "log.md"]);
const KEY = /^([A-Za-z_][A-Za-z0-9_-]*):[ \t]*(.*)$/;

function scalar(value) {
  value = value.trim();
  if (value.length >= 2 && value[0] === value[value.length - 1] && "\"'".includes(value[0])) return value.slice(1, -1);
  return value;
}

// The top-level front matter keys of a concept (strings, or lists of strings) and its Markdown body: [meta, body].
export function frontMatter(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  if (!lines.length || lines[0].trim() !== "---") return [{}, text];
  let end = -1;
  for (let i = 1; i < lines.length; i++) if (lines[i].trim() === "---") { end = i; break; }
  if (end < 0) return [{}, text];
  const meta = {};
  let key = null;
  let block = [];
  const close = () => {
    if (key === null || key in meta) return;
    const items = block.map((b) => b.trim()).filter(Boolean);
    const nested = (i) => i.slice(2).includes(":") && !"\"'".includes(i.slice(2).trim().slice(0, 1));
    if (items.length && items.every((i) => i.startsWith("- ")) && !items.some(nested)) {
      meta[key] = items.map((i) => scalar(i.slice(2)));
    } else if (items.length && !items.some((i) => KEY.test(i) || i.startsWith("- "))) {
      meta[key] = items.join(" "); // a folded or literal block scalar
    }
  };
  for (const line of lines.slice(1, end)) {
    const m = KEY.exec(line);
    if (m && !/^\s/.test(line)) {
      close();
      key = m[1];
      block = [];
      const value = m[2].trim();
      if ([">", "|", ">-", "|-", ">+", "|+", ""].includes(value)) continue;
      if (value.startsWith("[") && value.endsWith("]")) {
        meta[key] = value.slice(1, -1).split(",").filter((v) => v.trim()).map(scalar);
      } else if (!value.startsWith("{")) {
        meta[key] = scalar(value);
      }
    } else if (key !== null) {
      block.push(line);
    }
  }
  close();
  return [meta, lines.slice(end + 1).join("\n").replace(/^\n+/, "")];
}

// Unee documents ({ title, text }) for the concept files of a bundle, given as { path in the bundle: file text }.
export function okfDocuments(files, { includeDeprecated = false } = {}) {
  const out = [];
  for (const path of Object.keys(files).sort()) {
    const parts = path.replace(/\\/g, "/").replace(/^\/+|\/+$/g, "").split("/");
    const name = parts[parts.length - 1];
    if (!name.toLowerCase().endsWith(".md") || RESERVED.has(name.toLowerCase())) continue;
    const [meta, body] = frontMatter(files[path]);
    if (String(meta.status ?? "").toLowerCase() === "deprecated" && !includeDeprecated) continue;
    const title = typeof meta.title === "string" && meta.title ? meta.title : name.slice(0, -3).replace(/[_-]/g, " ");
    const where = parts.slice(0, -1).join(" / ");
    const head = typeof meta.description === "string" && meta.description ? [meta.description] : [];
    if (Array.isArray(meta.tags) && meta.tags.length) head.push("Tags: " + meta.tags.join(", "));
    const text = [...head, body.trim()].join("\n\n").trim();
    if (text) out.push({ title: where ? `${where} / ${title}` : title, text });
  }
  return out;
}
