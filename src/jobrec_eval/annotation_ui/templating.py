"""Tiny stdlib template renderer for the annotation web UI, in Jinja-compatible syntax.

Why this module exists instead of :mod:`jinja2`. The plan was ``Jinja2Templates``, but Jinja2
is NOT installed in this environment and is not a declared dependency of this project (checked
with ``import jinja2`` before anything was written; ``fastapi``/``starlette`` do not pull it
in). Installing it would add a third-party package to a reproduction package that must stay
runnable from the declared dependency set, so the templates are rendered here instead.

The templates under ``templates/`` are written in the Jinja SUBSET this renderer implements,
which is deliberately small:

- ``{{ path }}`` -- HTML-escaped interpolation; ``path`` is dotted (``item.job.title``) and may
  index a sequence by number (``turns.0.candidate_utterance``);
- ``{{ path | raw }}`` -- interpolation WITHOUT escaping, used only to drop an already-rendered
  page body into ``base.html``;
- ``{% for name in path %} ... {% endfor %}`` -- nesting allowed;
- ``{% if path %} ... {% else %} ... {% endif %}`` / ``{% if not path %}`` -- truthiness only.

There is no expression evaluation, no arithmetic, no filters beyond ``raw``/``e`` and no
arbitrary attribute calls, which is also why a template cannot execute anything: everything a
screen needs is precomputed into plain dicts and lists by
:mod:`~jobrec_eval.annotation_ui.views`.

What "subset" does and does not buy. The template SYNTAX above is a subset of Jinja's, so the
template files parse under Jinja; the swap is NOT a drop-in one, because this renderer's
semantics differ from a bare ``jinja2.Environment()`` in two ways that fail silently:

- **escaping**: escaping is on here by default, while ``jinja2.Environment()`` autoescapes
  NOTHING, so a swap must pass ``autoescape=select_autoescape(["html"])`` or every candidate
  utterance and job description stops being escaped -- a markup-injection regression on a
  rater's screen, with no error to notice;
- **boolean spelling**: ``{{ flag }}`` renders ``true``/``false`` here (see :func:`_stringify`)
  and ``True``/``False`` under Jinja, and ``static/annotate.js`` tests
  ``root.getAttribute('data-was-saved') === 'true'``, so a swap must re-check every boolean
  that reaches a template or the aria-live "saved" announcement quietly stops firing.

So swapping the engine means changing this module, setting autoescape explicitly, and
auditing the boolean spellings the JS depends on.

Escaping is on by default (``html.escape`` with quotes), so a candidate utterance, a job
description or a claim sentence containing ``<`` cannot inject markup into a rater's screen.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Splits a template into literal text and ``{{ ... }}`` / ``{% ... %}`` tokens.
_TOKEN_RE = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.DOTALL)
_VAR_RE = re.compile(r"^\{\{\s*(?P<path>[A-Za-z_][\w.]*)\s*(?:\|\s*(?P<filter>[a-z]+)\s*)?\}\}$")
_TAG_RE = re.compile(r"^\{%\s*(?P<body>.+?)\s*%\}$", re.DOTALL)
_FOR_RE = re.compile(r"^for\s+(?P<name>[A-Za-z_]\w*)\s+in\s+(?P<path>[A-Za-z_][\w.]*)$")
_IF_RE = re.compile(r"^if\s+(?P<negate>not\s+)?(?P<path>[A-Za-z_][\w.]*)$")

#: Filters this renderer understands. ``raw`` is the only escape-suppressing one and is used
#: exactly once (the page body inside ``base.html``).
_FILTERS = frozenset({"e", "escape", "raw"})

_MISSING = object()


class TemplateError(RuntimeError):
    """A template used syntax outside the supported subset, or a tag was left unclosed."""


@dataclass(frozen=True)
class _Text:
    text: str


@dataclass(frozen=True)
class _Var:
    path: str
    raw: bool = False


@dataclass(frozen=True)
class _For:
    name: str
    path: str
    body: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _If:
    path: str
    negate: bool = False
    body: tuple[Any, ...] = ()
    orelse: tuple[Any, ...] = ()


@dataclass
class _Cursor:
    tokens: list[str]
    index: int = 0
    #: Names of the block tags currently open, for a useful "unclosed block" message.
    open_blocks: list[str] = field(default_factory=list)

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.index]
        self.index += 1
        return token


def _resolve(path: str, context: Mapping[str, Any]) -> Any:
    """Look up a dotted path in the context; missing -> :data:`_MISSING`.

    Mapping keys, object attributes and numeric sequence indices are all accepted so a view
    model can be plain dicts (which it is) or dataclasses without the template caring.
    """
    parts = path.split(".")
    current: Any = context
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not part.isdigit() or int(part) >= len(current):
                return _MISSING
            current = current[int(part)]
        else:
            if not hasattr(current, part):
                return _MISSING
            current = getattr(current, part)
    return current


def _stringify(value: Any) -> str:
    if value is _MISSING or value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _truthy(value: Any) -> bool:
    return False if value is _MISSING else bool(value)


def _parse_nodes(cursor: _Cursor, *, stop: tuple[str, ...] = ()) -> tuple[Any, ...]:
    nodes: list[Any] = []
    while True:
        token = cursor.peek()
        if token is None:
            if stop:
                raise TemplateError(
                    f"unclosed {{% {cursor.open_blocks[-1]} %}} block; expected one of "
                    f"{', '.join(stop)}")
            return tuple(nodes)
        tag = _TAG_RE.match(token) if token.startswith("{%") else None
        if tag is not None:
            body = tag.group("body").strip()
            keyword = body.split()[0]
            if keyword in stop:
                return tuple(nodes)
            cursor.take()
            nodes.append(_parse_block(body, keyword, cursor))
            continue
        cursor.take()
        if token.startswith("{{"):
            match = _VAR_RE.match(token)
            if match is None:
                raise TemplateError(
                    f"unsupported interpolation {token!r}; only '{{{{ dotted.path }}}}' "
                    f"optionally with '| raw' is supported")
            filter_name = match.group("filter")
            if filter_name is not None and filter_name not in _FILTERS:
                raise TemplateError(
                    f"unknown filter {filter_name!r}; supported: {sorted(_FILTERS)}")
            nodes.append(_Var(path=match.group("path"), raw=filter_name == "raw"))
        else:
            nodes.append(_Text(token))


def _parse_block(body: str, keyword: str, cursor: _Cursor) -> Any:
    if keyword == "for":
        match = _FOR_RE.match(body)
        if match is None:
            raise TemplateError(f"unsupported for tag {{% {body} %}}")
        cursor.open_blocks.append(body)
        loop_body = _parse_nodes(cursor, stop=("endfor",))
        _expect(cursor, "endfor")
        cursor.open_blocks.pop()
        return _For(name=match.group("name"), path=match.group("path"), body=loop_body)
    if keyword == "if":
        match = _IF_RE.match(body)
        if match is None:
            raise TemplateError(f"unsupported if tag {{% {body} %}}")
        cursor.open_blocks.append(body)
        then_body = _parse_nodes(cursor, stop=("else", "endif"))
        else_body: tuple[Any, ...] = ()
        if _peek_keyword(cursor) == "else":
            _expect(cursor, "else")
            else_body = _parse_nodes(cursor, stop=("endif",))
        _expect(cursor, "endif")
        cursor.open_blocks.pop()
        return _If(path=match.group("path"), negate=bool(match.group("negate")),
                   body=then_body, orelse=else_body)
    raise TemplateError(f"unsupported tag {{% {body} %}}")


def _peek_keyword(cursor: _Cursor) -> str | None:
    token = cursor.peek()
    if token is None or not token.startswith("{%"):
        return None
    tag = _TAG_RE.match(token)
    return tag.group("body").strip().split()[0] if tag else None


def _expect(cursor: _Cursor, keyword: str) -> None:
    if _peek_keyword(cursor) != keyword:
        raise TemplateError(f"expected {{% {keyword} %}}, found {cursor.peek()!r}")
    cursor.take()


def parse(source: str) -> tuple[Any, ...]:
    """Parse template source into a node tree. Raises :class:`TemplateError` on bad syntax."""
    tokens = [token for token in _TOKEN_RE.split(source) if token != ""]
    return _parse_nodes(_Cursor(tokens))


def render_nodes(nodes: Sequence[Any], context: Mapping[str, Any]) -> str:
    """Render a parsed node tree against a context mapping."""
    out: list[str] = []
    for node in nodes:
        if isinstance(node, _Text):
            out.append(node.text)
        elif isinstance(node, _Var):
            value = _stringify(_resolve(node.path, context))
            out.append(value if node.raw else html.escape(value, quote=True))
        elif isinstance(node, _For):
            sequence = _resolve(node.path, context)
            if sequence is _MISSING or sequence is None:
                continue
            if isinstance(sequence, Mapping) or not isinstance(sequence, Sequence):
                raise TemplateError(
                    f"{node.path!r} is not a list; '{{% for %}}' iterates lists only")
            for element in sequence:
                out.append(render_nodes(node.body, {**context, node.name: element}))
        elif isinstance(node, _If):
            hit = _truthy(_resolve(node.path, context))
            if node.negate:
                hit = not hit
            out.append(render_nodes(node.body if hit else node.orelse, context))
        else:  # pragma: no cover - node types are closed
            raise TemplateError(f"unknown node {node!r}")
    return "".join(out)


class TemplateRenderer:
    """Renders ``*.html`` files from one directory, with an mtime-checked parse cache."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"template directory {self.directory} does not exist")
        self._cache: dict[str, tuple[float, tuple[Any, ...]]] = {}

    def _nodes(self, name: str) -> tuple[Any, ...]:
        path = self.directory / name
        if not path.is_file():
            raise FileNotFoundError(f"no template {name!r} in {self.directory}")
        mtime = path.stat().st_mtime
        cached = self._cache.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        nodes = parse(path.read_text(encoding="utf-8"))
        self._cache[name] = (mtime, nodes)
        return nodes

    def render(self, name: str, context: Mapping[str, Any] | None = None) -> str:
        """Render one template file."""
        return render_nodes(self._nodes(name), dict(context or {}))

    def render_page(self, name: str, context: Mapping[str, Any] | None = None, *,
                    base: str = "base.html") -> str:
        """Render a page template and wrap it in ``base.html`` as ``{{ content | raw }}``.

        Template inheritance (``{% extends %}``) is not part of the supported subset; wrapping
        the rendered body is the same result with none of the engine complexity.
        """
        values = dict(context or {})
        body = self.render(name, values)
        return self.render(base, {**values, "content": body})
