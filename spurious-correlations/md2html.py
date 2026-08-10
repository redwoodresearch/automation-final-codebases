#!/usr/bin/env python3
"""Minimal, dependency-free Markdown -> HTML for copy-pasting into Google Docs.
Handles: headings, GFM tables, bold/italic, inline code, links, blockquotes,
ordered/unordered lists, fenced code blocks, horizontal rules, paragraphs.
Output is styled so a browser render pastes into Docs with real headings + tables."""
import base64
import html
import os
import re
import sys

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _img_data_uri(src):
    """Embed a local figure (matched by basename in figs/) as a base64 data URI."""
    path = os.path.join(FIGDIR, os.path.basename(src))
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def inline(text):
    # protect math spans (keep raw LaTeX for MathJax) before escaping / markup
    maths = []
    def _math(m):
        maths.append(m.group(0))
        return f"\x01{len(maths)-1}\x01"
    text = re.sub(r"\$\$.+?\$\$", _math, text, flags=re.S)
    text = re.sub(r"\$[^$\n]+?\$", _math, text)
    # escape first, then re-introduce markup
    text = html.escape(text, quote=False)
    # inline code (protect contents)
    codes = []
    def _code(m):
        codes.append(m.group(1))
        return f"\x00{len(codes)-1}\x00"
    text = re.sub(r"`([^`]+)`", _code, text)
    # images ![alt](src) -> inline base64 <img> (must run before links)
    def _img(m):
        alt, src = m.group(1), m.group(2)
        uri = _img_data_uri(src)
        if uri is None:
            return f'<em>[missing figure: {html.escape(src)}]</em>'
        return f'<img alt="{html.escape(alt)}" src="{uri}" style="max-width:100%;height:auto">'
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img, text)
    # links [t](u)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # bold then italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<em>\1</em>", text)
    # restore code
    for i, c in enumerate(codes):
        text = text.replace(f"\x00{i}\x00", f"<code>{html.escape(c, quote=False)}</code>")
    # restore math (raw LaTeX for MathJax to render)
    for i, mth in enumerate(maths):
        text = text.replace(f"\x01{i}\x01", mth)
    return text


def is_table_sep(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def render(md):
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # fenced code
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        # table: current line has pipes and next line is a separator
        if "|" in line and i + 1 < n and is_table_sep(lines[i + 1]):
            def cells(row):
                r = row.strip()
                if r.startswith("|"):
                    r = r[1:]
                if r.endswith("|"):
                    r = r[:-1]
                return [c.strip() for c in r.split("|")]
            header = cells(line)
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i]))
                i += 1
            t = ['<table>', '<thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in header) + '</tr></thead>', '<tbody>']
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("\n".join(t))
            continue
        # raw passthrough for collapsible dropdowns (details/summary)
        st = line.strip()
        if st in ("<details>", "</details>", "</summary>") or st.startswith("<summary>"):
            out.append(st)
            i += 1
            continue
        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        # horizontal rule
        if re.match(r"^\s*(\*\s*){3,}$", line) or re.match(r"^\s*(-\s*){3,}$", line):
            out.append("<hr>")
            i += 1
            continue
        # blockquote
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        # unordered list (gather wrapped continuation lines into the same <li>)
        if re.match(r"^\s*[-*+]\s+", line):
            items = []
            while i < n:
                if re.match(r"^\s*[-*+]\s+", lines[i]):
                    items.append(re.sub(r"^\s*[-*+]\s+", "", lines[i]))
                    i += 1
                elif items and lines[i].strip() and not re.match(r"^(#{1,6}\s|>|```|\s*\d+\.\s)", lines[i]):
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            out.append("<ul>" + "".join("<li>" + inline(it) + "</li>" for it in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < n:
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                    i += 1
                elif items and lines[i].strip() and not re.match(r"^(#{1,6}\s|>|```|\s*[-*+]\s)", lines[i]):
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            out.append("<ol>" + "".join("<li>" + inline(it) + "</li>" for it in items) + "</ol>")
            continue
        # blank
        if not line.strip():
            i += 1
            continue
        # paragraph: gather until blank / block start
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*[-*+]\s|\s*\d+\.\s|>|```)", lines[i]) and not ("|" in lines[i] and i + 1 < n and is_table_sep(lines[i + 1])):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


CSS = """body{font-family:Arial,Helvetica,sans-serif;line-height:1.5;max-width:820px;margin:2rem auto;padding:0 1rem;color:#111}
h1{font-size:1.8rem}h2{font-size:1.4rem;margin-top:1.6rem}h3{font-size:1.15rem}
table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #999;padding:6px 10px;text-align:left;vertical-align:top;font-size:0.95rem}
th{background:#f0f0f0}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-family:monospace}
pre{background:#f4f4f4;padding:10px;overflow:auto}
blockquote{border-left:3px solid #ccc;margin:1rem 0;padding:0.2rem 1rem;color:#555}"""


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    md = open(src, encoding="utf-8").read()
    body = render(md)
    # --no-mathjax: leave $...$ as literal dollar-sign text (for pasting into LessWrong, whose
    # editor renders $...$ itself; MathJax here would pre-render them and eat the dollar signs).
    mathjax = "" if "--no-mathjax" in sys.argv else (
        "<script>window.MathJax={tex:{inlineMath:[['$','$']]}};</script>"
        "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>")
    doc = f"<!doctype html>\n<html><head><meta charset='utf-8'><style>{CSS}</style>{mathjax}</head>\n<body>\n{body}\n</body></html>\n"
    open(dst, "w", encoding="utf-8").write(doc)
    print(f"wrote {dst} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
