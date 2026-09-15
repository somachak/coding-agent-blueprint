"""
build_site.py - turn docs/course.html into the finished site in site/.

Why a build step: the course must show the REAL code, never a pasted copy
that drifts. So the narrative page contains placeholders like

    {{code:course/ch01_one_call.py}}
    {{code:agent/loop.py:1-40}}

and this script replaces each one with the current file contents,
syntax-highlighted, at build time. It also copies the chat page into
site/app/ so one static folder holds everything Cloudflare serves.

    python3 docs/build_site.py
"""

import html
import io
import re
import shutil
import tokenize
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / "docs" / "course.html"
SITE = PROJECT / "site"

PLACEHOLDER = re.compile(r"\{\{code:([^:}]+)(?::(\d+)-(\d+))?\}\}")


def highlight_python(source: str, first_line: int = 1) -> str:
    """Wrap comments, docstrings, strings, keywords and names in spans.

    Uses Python's own tokenizer, so a '#' inside a string is never treated
    as a comment and a triple-quoted string is always a string.
    """
    keywords = {
        "def", "class", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or",
        "import", "from", "as", "try", "except", "finally", "raise", "with", "yield", "lambda",
        "pass", "break", "continue", "is", "None", "True", "False", "global", "nonlocal", "assert", "del",
    }
    builtins = {"print", "len", "str", "int", "dict", "list", "set", "open", "range", "isinstance", "json"}
    out = []
    position = (1, 0)
    lines = source.splitlines(keepends=True)

    def line_at(row):
        """The text of a 1-based line, or '' for the tokenizer's phantom line after the end."""
        return lines[row - 1] if 1 <= row <= len(lines) else ""

    def slice_between(start, end):
        (srow, scol), (erow, ecol) = start, end
        if srow == erow:
            return line_at(srow)[scol:ecol]
        text = line_at(srow)[scol:]
        for row in range(srow + 1, erow):
            text += line_at(row)
        return text + line_at(erow)[:ecol]

    prev_significant = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            gap = slice_between(position, token.start)
            out.append(html.escape(gap))
            text = html.escape(token.string)
            kind = token.type
            css = None
            if kind == tokenize.COMMENT:
                css = "c"
            elif kind == tokenize.STRING:
                is_docstring = (
                    token.string.startswith(('"""', "'''"))
                    and prev_significant in (None, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL, "colon")
                )
                css = "d" if is_docstring else "s"
            elif kind == tokenize.NAME and token.string in keywords:
                css = "k"
            elif kind == tokenize.NAME and token.string in builtins:
                css = "b"
            elif kind == tokenize.NUMBER:
                css = "n"
            out.append(f'<span class="{css}">{text}</span>' if css else text)
            position = token.end
            if kind == tokenize.OP and token.string == ":":
                prev_significant = "colon"
            elif kind not in (tokenize.NL, tokenize.COMMENT):
                prev_significant = kind
    except tokenize.TokenError:
        return html.escape(source)
    out.append(html.escape(slice_between(position, (len(lines) + 1, 0))))
    body = "".join(out)
    numbered = []
    for offset, line in enumerate(body.split("\n")):
        numbered.append(f'<span class="ln">{first_line + offset:4d}</span>{line}')
    return "\n".join(numbered)


def render_code(path: str, start: int | None, end: int | None) -> str:
    file = PROJECT / path
    text = file.read_text(encoding="utf-8")
    label = path
    if start is not None:
        lines = text.splitlines(keepends=True)
        text = "".join(lines[start - 1:end])
        label = f"{path} (lines {start}-{end})"
        first = start
    else:
        first = 1
    if text.endswith("\n"):
        text = text[:-1]
    body = highlight_python(text, first) if path.endswith(".py") else html.escape(text)
    return (
        f'<figure class="code"><figcaption><span class="path">{html.escape(label)}</span>'
        f'<span class="hint">real file, embedded at build time</span></figcaption>'
        f'<pre><code>{body}</code></pre></figure>'
    )


def build() -> None:
    page = SOURCE.read_text(encoding="utf-8")

    def replace(match: re.Match) -> str:
        path = match.group(1)
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else None
        return render_code(path, start, end)

    page = PLACEHOLDER.sub(replace, page)
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / "app").mkdir(exist_ok=True)
    shutil.copy(PROJECT / "web" / "index.html", SITE / "app" / "index.html")
    print(f"built {SITE / 'index.html'} ({len(page)} bytes) and site/app/index.html")


if __name__ == "__main__":
    build()
