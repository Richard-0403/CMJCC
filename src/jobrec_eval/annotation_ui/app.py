"""Localhost web UI for the two-rater human annotation pass (checklist items 10/11).

.. warning::

   **THIS APP HAS NO AUTHENTICATION.** The rater cookie is an IDENTITY for attributing labels,
   not an access control: anybody who can reach the port can pick any rater from the pool and
   write labels as them. It is built for the situation the annotation pass actually runs in --
   two or three people sharing one machine, one at a time, over ``127.0.0.1`` -- and the
   default host is therefore loopback only. Do NOT bind it to a routable interface, do not put
   it behind a tunnel, and do not expose it to a network you do not control. If you need remote
   access, put a real authenticating reverse proxy in front of it; nothing here is a substitute
   for one.

This module is a PRESENTATION layer. Every read and write goes through
:class:`~jobrec_eval.annotation_ui.store.AnnotationStore`, which is where rater isolation and
blinding are enforced, so a bug in a template or a route cannot break either invariant. The
routes are split into two groups and the split is the auditable boundary:

**Rater-facing routes** (``/``, ``/rater``, ``/queue``, ``/annotate/{kind}``, ``/progress``,
``/api/session``, ``/api/queue``, ``/api/item``, ``/api/annotations``, ``/api/progress``) call
ONLY the rater-scoped store methods: :meth:`~store.AnnotationStore.raters`,
:meth:`~store.AnnotationStore.queue`, :meth:`~store.AnnotationStore.next_item`,
:meth:`~store.AnnotationStore.rater_item`, :meth:`~store.AnnotationStore.annotation`,
:meth:`~store.AnnotationStore.upsert_annotation` and
:meth:`~store.AnnotationStore.progress`. Each of those takes a ``rater_id`` and filters on it,
and the ``rater_id`` always comes from the signed session cookie -- never from a body, query
string or header -- so a request cannot address another rater's queue at all. None of these
routes calls :meth:`~store.AnnotationStore.disagreements` or
:meth:`~store.AnnotationStore.iter_export_records`, which are the only methods that carry the
other rater's label, the oracle grade or the validator verdict.

**Adjudicator/export routes** (``/adjudication``, ``/export``, ``/api/adjudication/*``,
``/api/export``) are the only ones that touch that analysis-side data, and no rater page links
to them: an adjudicator reaches them by typing the URL. Showing a rater either the machine's
answer or their colleague's label would turn an independent judgement into agreement with what
they were shown, and the reported Cohen's kappa would no longer measure anything.

Dependency notes (checked in this environment before writing a line of it):

- ``jinja2`` is NOT installed and is not a declared dependency, so templates are rendered by
  :mod:`~jobrec_eval.annotation_ui.templating`, a small stdlib renderer that consumes the same
  Jinja syntax subset the templates are written in;
- ``python-multipart`` is NOT installed, so there are no HTML form posts and no
  ``fastapi.Form`` parameters. Every write is a JSON ``fetch`` to a JSON endpoint, which is
  also what lets the client post the measured ``duration_ms``;
- ``itsdangerous`` is NOT installed, so the session cookie is signed here with
  :mod:`hmac` + SHA-256 instead of via Starlette's ``SessionMiddleware``.

No third-party package was added for any of the three.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from .export import export_annotations
from .store import (
    DB_FILENAME,
    KIND_CLAIM,
    KIND_RELEVANCE,
    KINDS,
    LABEL_RANGES,
    AnnotationStore,
    InvalidLabelError,
    NotAssignedError,
    UnknownRaterError,
    open_store,
)
from .templating import TemplateRenderer
from .views import FLAG_UNRESOLVABLE_EVIDENCE, item_view

#: Loopback by default. See the module warning: this app must not be bound anywhere else
#: without an authenticating proxy in front of it.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Session cookie names. Two separate cookies, so a rater's identity cannot be used to record
#: an adjudication and an adjudicator's cannot be used to label.
RATER_COOKIE = "cmjcc_rater"
ADJUDICATOR_COOKIE = "cmjcc_adjudicator"

#: Shown on every page and repeated in the serve command's help.
NO_AUTH_NOTICE = (
    "No login: choosing a rater records WHO a label belongs to, it does not protect anything. "
    "This tool is meant to run on 127.0.0.1 on a shared annotation machine.")

#: Human wording for the two item kinds.
KIND_LABELS = {KIND_RELEVANCE: "relevance", KIND_CLAIM: "grounding"}

_TEMPLATES_DIR = Path(__file__).with_name("templates")
_STATIC_DIR = Path(__file__).with_name("static")


# --------------------------------------------------------------------------- signed cookie
def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_cookie(payload: dict[str, Any], secret: str) -> str:
    """Serialise and sign a cookie payload as ``base64(json).base64(hmac-sha256)``.

    Signed so a rater id in a cookie cannot simply be typed by hand into the browser's
    developer tools and passed off as somebody else's attribution. It is NOT a password: the
    holder of a valid cookie is trusted because the app only listens on loopback.
    """
    body = _b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64encode(digest)}"


def read_cookie(token: str | None, secret: str) -> dict[str, Any] | None:
    """Verify and decode a cookie written by :func:`sign_cookie`; ``None`` if it is not valid.

    Any failure -- missing, malformed, tampered signature, non-object payload -- returns
    ``None`` and the caller treats the visitor as having chosen nobody yet.
    """
    if not token or "." not in token:
        return None
    body, _, signature = token.rpartition(".")
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_b64decode(signature), expected):
            return None
        payload = json.loads(_b64decode(body))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_loopback_host(host: str) -> bool:
    """True for ``localhost``/``127.0.0.0/8``/``::1``; used to gate the ``--host`` flag."""
    import ipaddress

    normalized = (host or "").strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


# ------------------------------------------------------------------------- request bodies
class SelectRaterRequest(BaseModel):
    """Which registered rater is sitting at the machine right now."""

    rater_id: str = Field(min_length=1)


class SaveAnnotationRequest(BaseModel):
    """One rater's answer for one item.

    There is deliberately NO ``rater_id`` field: the rater comes from the signed session
    cookie, so a request body cannot nominate whose label it is. ``duration_ms`` is measured
    by the browser from item render to save and is the only source for reported annotation
    effort, so it travels with every write.
    """

    item_key: str = Field(min_length=1)
    label: int
    notes: str = ""
    flags: str = ""
    duration_ms: int | None = Field(default=None, ge=0)


class AdjudicationRequest(BaseModel):
    """An adjudicator's final verdict on one disagreement, with the required reason."""

    item_key: str = Field(min_length=1)
    final_label: int
    reason: str = ""
    adjudicator: str = ""


class AdjudicatorRequest(BaseModel):
    """Who is adjudicating; recorded for attribution exactly like a rater id."""

    adjudicator: str = Field(min_length=1)


def create_app(annotation_dir: str | Path, *, secret_key: str | None = None,
               export_dir: str | Path | None = None,
               release_dir: str | Path | None = None,
               templates_dir: str | Path | None = None,
               static_dir: str | Path | None = None) -> FastAPI:
    """Build the annotation web app over an existing annotation store.

    Args:
        annotation_dir: Directory holding ``annotation.sqlite3`` (built by
            ``python -m jobrec_eval.annotation_ui build``). The store is opened per request
            with ``create=False``: the UI never creates or migrates a store, so pointing it at
            the wrong directory fails loudly instead of quietly starting an empty pass.
        secret_key: HMAC key for the identity cookies. Defaults to a fresh random key, which
            means restarting the server asks whoever is sitting there to pick their name again
            -- harmless, and it avoids shipping a hard-coded key.
        export_dir: Where ``/export`` writes the two CSVs. Defaults to
            ``<annotation_dir>/export``. Fixed at app creation, so the export endpoint takes no
            path from the request and cannot be pointed at an arbitrary location.
        release_dir: Where the JSONL dump and manifest go. Defaults to
            ``<export_dir>/human_annotations``.
        templates_dir / static_dir: Overridable for tests; default to the package's own.

    Raises:
        FileNotFoundError: No annotation store in ``annotation_dir``.
    """
    directory = Path(annotation_dir)
    if not (directory / DB_FILENAME).is_file():
        raise FileNotFoundError(
            f"no annotation store at {directory / DB_FILENAME}; build one first with "
            f"python -m jobrec_eval.annotation_ui build --annotation-dir {directory} ...")
    secret = secret_key or secrets.token_urlsafe(32)
    exports = Path(export_dir) if export_dir is not None else directory / "export"
    releases = Path(release_dir) if release_dir is not None else exports / "human_annotations"
    templates = TemplateRenderer(templates_dir or _TEMPLATES_DIR)
    static_root = Path(static_dir or _STATIC_DIR)

    app = FastAPI(title="CMJCC human annotation UI", version="1.0.0",
                  description=NO_AUTH_NOTICE)
    app.mount("/static", StaticFiles(directory=static_root), name="static")
    app.state.annotation_dir = directory
    app.state.export_dir = exports
    app.state.release_dir = releases
    app.state.secret_key = secret

    @contextmanager
    def store_session() -> Iterator[AnnotationStore]:
        """One SQLite connection per request (WAL + busy timeout make that safe)."""
        store = open_store(directory, create=False)
        try:
            yield store
        finally:
            store.close()

    # ----------------------------------------------------------------- identity helpers
    def session_rater(request: Request, store: AnnotationStore) -> str | None:
        """The rater id from the signed cookie, if it is still a registered rater."""
        payload = read_cookie(request.cookies.get(RATER_COOKIE), secret)
        if not payload:
            return None
        rater_id = str(payload.get("rater_id") or "")
        return rater_id if rater_id in store.raters() else None

    def require_rater(request: Request, store: AnnotationStore) -> str:
        """The session rater, or 403. Writes always resolve the rater THIS way."""
        rater_id = session_rater(request, store)
        if rater_id is None:
            raise HTTPException(status_code=403,
                                detail="no rater selected in this session; choose one at /rater")
        return rater_id

    def session_adjudicator(request: Request) -> str:
        payload = read_cookie(request.cookies.get(ADJUDICATOR_COOKIE), secret)
        return str((payload or {}).get("adjudicator") or "")

    # ------------------------------------------------------------------ page rendering
    def rater_nav() -> list[dict[str, str]]:
        """Rater navigation. Contains no adjudication or export link, by design."""
        return [
            {"href": "/queue", "text": "My queue"},
            {"href": f"/annotate/{KIND_RELEVANCE}", "text": "Relevance items"},
            {"href": f"/annotate/{KIND_CLAIM}", "text": "Grounding items"},
            {"href": "/progress", "text": "My progress"},
            {"href": "/rater", "text": "Switch rater"},
        ]

    def page(name: str, *, title: str, rater_id: str | None = None,
             nav: list[dict[str, str]] | None = None, script: str = "",
             status_message: str = "", status_code: int = 200,
             **context: Any) -> HTMLResponse:
        html = templates.render_page(name, {
            "title": title,
            "rater_id": rater_id or "",
            "nav": nav if nav is not None else [],
            "script": script,
            "status_message": status_message,
            "no_auth_notice": NO_AUTH_NOTICE,
            **context,
        })
        return HTMLResponse(html, status_code=status_code)

    def progress_payload(store: AnnotationStore, rater_id: str) -> dict[str, Any]:
        """Rater-scoped progress numbers, shared by the page and the JSON endpoint."""
        progress = store.progress(rater_id)
        median = progress.median_duration_ms
        per_kind = {}
        for kind in KINDS:
            items = store.queue(rater_id, kind=kind)
            done = sum(1 for item in items if item.done)
            per_kind[kind] = {"assigned": len(items), "completed": done,
                              "remaining": len(items) - done}
        return {
            "rater_id": rater_id,
            "assigned": progress.assigned,
            "completed": progress.completed,
            "remaining": progress.remaining,
            "percent_complete": round(progress.fraction_complete * 100, 1),
            "median_duration_ms": median,
            "median_seconds": None if median is None else round(median / 1000, 1),
            "total_duration_ms": progress.total_duration_ms,
            "per_kind": per_kind,
        }

    def annotate_url(kind: str, item_key: str | None = None, *, after: str | None = None,
                     saved: bool = False) -> str:
        url = f"/annotate/{kind}"
        params = []
        if item_key:
            params.append(f"item={quote(item_key, safe='')}")
        if after:
            params.append(f"after={quote(after, safe='')}")
        if saved:
            params.append("saved=1")
        return f"{url}?{'&'.join(params)}" if params else url

    # ---------------------------------------------------------------------- health
    @app.get("/health/live")
    def live() -> dict:
        """Liveness only; deliberately reports nothing about the annotation content."""
        return {"status": "live", "authentication": "none", "bind_default": DEFAULT_HOST}

    # ------------------------------------------------------------- rater selection
    @app.get("/", include_in_schema=False)
    def index(request: Request) -> Response:
        with store_session() as store:
            rater_id = session_rater(request, store)
        return RedirectResponse("/queue" if rater_id else "/rater", status_code=303)

    @app.get("/rater", response_class=HTMLResponse)
    def rater_selection(request: Request) -> HTMLResponse:
        """Pick who you are. Store method: ``raters()`` (the registered pool only)."""
        with store_session() as store:
            current = session_rater(request, store)
            raters = [{"rater_id": rater, "current": rater == current}
                      for rater in store.raters()]
        return page("rater_select.html", title="Choose your rater name", rater_id=current,
                    nav=rater_nav() if current else [], script="/static/rater.js",
                    raters=raters, has_current=bool(current), current_rater=current or "")

    @app.post("/api/session")
    def select_rater(payload: SelectRaterRequest) -> JSONResponse:
        """Record the identity in a signed, HTTP-only cookie. Store method: ``raters()``."""
        with store_session() as store:
            if payload.rater_id not in store.raters():
                raise HTTPException(status_code=400,
                                    detail=f"{payload.rater_id!r} is not a registered rater")
        response = JSONResponse({"rater_id": payload.rater_id, "next_url": "/queue"})
        response.set_cookie(
            RATER_COOKIE,
            sign_cookie({"rater_id": payload.rater_id,
                         "issued_at": datetime.now(UTC).isoformat()}, secret),
            httponly=True, samesite="lax", path="/")
        return response

    @app.post("/api/session/clear")
    def clear_rater() -> JSONResponse:
        """Switch rater: forget the identity so the next person picks their own name."""
        response = JSONResponse({"rater_id": None, "next_url": "/rater"})
        response.delete_cookie(RATER_COOKIE, path="/")
        return response

    # ------------------------------------------------------------------- rater queue
    @app.get("/queue", response_class=HTMLResponse)
    def queue_page(request: Request) -> Response:
        """This rater's queue. Store methods: ``queue(rater_id)``, ``progress(rater_id)``."""
        with store_session() as store:
            rater_id = session_rater(request, store)
            if rater_id is None:
                return RedirectResponse("/rater", status_code=303)
            items = [{
                "item_key": item.item_key,
                "kind": item.kind,
                "kind_label": KIND_LABELS.get(item.kind, item.kind),
                "position": item.position + 1,
                "status": "labelled" if item.done else "not labelled",
                "done": item.done,
                "label": "" if item.label is None else str(item.label),
                "url": annotate_url(item.kind, item.item_key),
            } for item in store.queue(rater_id)]
            progress = progress_payload(store, rater_id)
        return page("queue.html", title="My queue", rater_id=rater_id, nav=rater_nav(),
                    items=items, progress=progress, has_items=bool(items),
                    relevance_url=annotate_url(KIND_RELEVANCE),
                    claim_url=annotate_url(KIND_CLAIM))

    @app.get("/api/queue")
    def queue_json(request: Request, kind: str | None = Query(default=None)) -> dict:
        """Store method: ``queue(rater_id, kind=...)``. Rater-scoped, payload omitted."""
        _check_kind(kind)
        with store_session() as store:
            rater_id = require_rater(request, store)
            return {"rater_id": rater_id, "items": [
                {"item_key": item.item_key, "kind": item.kind, "position": item.position,
                 "done": item.done, "label": item.label}
                for item in store.queue(rater_id, kind=kind)]}

    # -------------------------------------------------------------- annotation screens
    @app.get("/annotate/{kind}", response_class=HTMLResponse)
    def annotate(request: Request, kind: str, item: str | None = Query(default=None),
                 after: str | None = Query(default=None),
                 saved: str | None = Query(default=None)) -> Response:
        """Render one item for THIS rater.

        Store methods: ``rater_item(rater_id, item_key)`` for an explicit item,
        ``queue(rater_id, include_done=False, kind=...)`` for "next" and for skipping. Both are
        rater-scoped, so an item belonging to the other rater raises ``NotAssignedError`` and
        is answered 404 -- never rendered.
        """
        _check_kind(kind, required=True)
        with store_session() as store:
            rater_id = session_rater(request, store)
            if rater_id is None:
                return RedirectResponse("/rater", status_code=303)
            try:
                target = _resolve_item(store, rater_id, kind, item, after)
            except NotAssignedError as exc:
                return page("not_assigned.html", title="Not your item", rater_id=rater_id,
                            nav=rater_nav(), status_code=404, detail=str(exc),
                            item_key=item or after or "")
            progress = progress_payload(store, rater_id)
            if target is None:
                return page("queue_complete.html", title="Queue complete", rater_id=rater_id,
                            nav=rater_nav(), progress=progress,
                            kind_label=KIND_LABELS.get(kind, kind))
            view = item_view(target)
            pending = store.queue(rater_id, include_done=False, kind=kind)
        template = ("annotate_relevance.html" if kind == KIND_RELEVANCE
                    else "annotate_claim.html")
        return page(
            template, title=f"{KIND_LABELS.get(kind, kind).title()} item", rater_id=rater_id,
            nav=rater_nav(), script="/static/annotate.js",
            status_message="", progress=progress,
            kind_label=KIND_LABELS.get(kind, kind),
            remaining_in_kind=len(pending),
            save_url="/api/annotations",
            skip_url=annotate_url(kind, after=target.item_key),
            queue_url="/queue",
            was_saved=saved == "1",
            **view)

    @app.get("/api/item")
    def item_json(request: Request, item: str = Query(...)) -> dict:
        """Store method: ``rater_item(rater_id, item_key)``. Carries the payload only.

        The payload is the blinded, rater-facing half of the item; ``RaterItem`` has no field
        for the analysis side, so there is nothing here to filter out.
        """
        with store_session() as store:
            rater_id = require_rater(request, store)
            record = store.rater_item(rater_id, item)
            return {"item_key": record.item_key, "kind": record.kind,
                    "position": record.position, "done": record.done, "label": record.label,
                    "notes": record.notes, "flags": record.flags, "payload": record.payload}

    @app.post("/api/annotations")
    def save_annotation(request: Request, payload: SaveAnnotationRequest) -> dict:
        """Save this rater's label. Store method: ``upsert_annotation(...)``.

        The rater id comes from the session cookie and the body has no field for it, so a
        request cannot write a label as somebody else. An unassigned item raises
        ``NotAssignedError`` (404) and an out-of-range label ``InvalidLabelError`` (422); both
        are refused by the store, not by this route.
        """
        with store_session() as store:
            rater_id = require_rater(request, store)
            record = store.upsert_annotation(
                payload.item_key, rater_id, payload.label, notes=payload.notes,
                flags=payload.flags, duration_ms=payload.duration_ms)
            kind = store.rater_item(rater_id, payload.item_key).kind
            pending = store.queue(rater_id, include_done=False, kind=kind)
            progress = progress_payload(store, rater_id)
        next_key = pending[0].item_key if pending else None
        return {
            "saved": True,
            "item_key": record.item_key,
            "label": record.label,
            "notes": record.notes,
            "flags": record.flags,
            "duration_ms": record.duration_ms,
            "revised": record.created_at != record.updated_at,
            "queue_complete": next_key is None,
            "next_url": (annotate_url(kind, next_key, saved=True) if next_key
                         else "/queue"),
            "progress": progress,
        }

    @app.get("/progress", response_class=HTMLResponse)
    def progress_page(request: Request) -> Response:
        """Store methods: ``progress(rater_id)``, ``queue(rater_id, kind=...)``."""
        with store_session() as store:
            rater_id = session_rater(request, store)
            if rater_id is None:
                return RedirectResponse("/rater", status_code=303)
            progress = progress_payload(store, rater_id)
        kinds = [{"kind": kind, "kind_label": KIND_LABELS.get(kind, kind), **counts}
                 for kind, counts in progress["per_kind"].items()]
        return page("progress.html", title="My progress", rater_id=rater_id, nav=rater_nav(),
                    progress=progress, kinds=kinds,
                    has_median=progress["median_seconds"] is not None,
                    relevance_url=annotate_url(KIND_RELEVANCE),
                    claim_url=annotate_url(KIND_CLAIM))

    @app.get("/api/progress")
    def progress_json(request: Request) -> dict:
        """Store methods: ``progress(rater_id)``, ``queue(rater_id, kind=...)``."""
        with store_session() as store:
            rater_id = require_rater(request, store)
            return progress_payload(store, rater_id)

    # ------------------------------------------------------- adjudication (NOT rater-facing)
    def adjudicator_nav() -> list[dict[str, str]]:
        return [{"href": "/adjudication", "text": "Adjudication queue"},
                {"href": "/export", "text": "Export"}]

    @app.get("/adjudication", response_class=HTMLResponse)
    def adjudication_page(request: Request,
                          open_only: str | None = Query(default=None)) -> HTMLResponse:
        """The disagreement worklist. Store method: ``disagreements(...)``.

        Reachable only by typing this URL: no rater page links here, because the page shows
        BOTH raters' labels side by side and seeing a colleague's label would destroy the
        independence the reported kappa rests on.
        """
        unadjudicated_only = open_only == "1"
        with store_session() as store:
            cases = store.disagreements(unadjudicated_only=unadjudicated_only)
            rows = [{
                "index": position,
                "item_key": case.item_key,
                "kind": case.kind,
                "kind_label": KIND_LABELS.get(case.kind, case.kind),
                "scenario_id": case.scenario_id or "",
                "job_id": case.job_id or "",
                "claim_id": case.claim_id or "",
                "slot_1_rater": case.slot_1_rater,
                "slot_2_rater": case.slot_2_rater,
                "slot_1_label": case.slot_1_label,
                "slot_2_label": case.slot_2_label,
                "adjudicated": case.adjudicated,
                "adjudicated_label": ("" if case.adjudicated_label is None
                                      else str(case.adjudicated_label)),
                "adjudicator": case.adjudicator or "",
                "status": "adjudicated" if case.adjudicated else "open",
                "options": [{"value": str(value)} for value in LABEL_RANGES[case.kind]],
            } for position, case in enumerate(cases, start=1)]
        return page("adjudication.html", title="Adjudication queue",
                    nav=adjudicator_nav(), script="/static/adjudication.js",
                    rows=rows, has_rows=bool(rows), open_only=unadjudicated_only,
                    adjudicator=session_adjudicator(request))

    @app.post("/api/adjudication/session")
    def select_adjudicator(payload: AdjudicatorRequest) -> JSONResponse:
        """Remember the adjudicator's name for attribution (identity, not authentication)."""
        response = JSONResponse({"adjudicator": payload.adjudicator})
        response.set_cookie(
            ADJUDICATOR_COOKIE,
            sign_cookie({"adjudicator": payload.adjudicator,
                         "issued_at": datetime.now(UTC).isoformat()}, secret),
            httponly=True, samesite="lax", path="/")
        return response

    @app.get("/api/adjudication/queue")
    def adjudication_queue_json(open_only: str | None = Query(default=None)) -> dict:
        """Store method: ``disagreements(...)``. Adjudicator-side only."""
        with store_session() as store:
            cases = store.disagreements(unadjudicated_only=open_only == "1")
            return {"count": len(cases), "cases": [
                {"item_key": case.item_key, "kind": case.kind,
                 "slot_1_rater": case.slot_1_rater, "slot_1_label": case.slot_1_label,
                 "slot_2_rater": case.slot_2_rater, "slot_2_label": case.slot_2_label,
                 "adjudicated_label": case.adjudicated_label,
                 "adjudicator": case.adjudicator} for case in cases]}

    @app.post("/api/adjudication/verdicts")
    def record_verdict(request: Request, payload: AdjudicationRequest) -> dict:
        """Store method: ``record_adjudication(item_key, adjudicator, final_label, reason)``.

        The reason is REQUIRED: an adjudicated grade that replaces two raters' judgements is
        reported in the thesis, so the written justification travels with it into the export's
        notes column and the archive dump. An empty reason is a 422, not a silent blank.
        """
        adjudicator = payload.adjudicator.strip() or session_adjudicator(request)
        if not adjudicator:
            raise HTTPException(status_code=422,
                                detail="an adjudicator name is required for attribution")
        if not payload.reason.strip():
            raise HTTPException(
                status_code=422,
                detail="a written reason is required: it is reported with the adjudicated label")
        with store_session() as store:
            verdict = store.record_adjudication(payload.item_key, adjudicator,
                                                payload.final_label, payload.reason.strip())
            open_cases = len(store.disagreements(unadjudicated_only=True))
        return {"item_key": verdict.item_key, "final_label": verdict.final_label,
                "adjudicator": verdict.adjudicator, "reason": verdict.reason,
                "created_at": verdict.created_at, "open_cases": open_cases}

    # ------------------------------------------------------------ export (NOT rater-facing)
    @app.get("/export", response_class=HTMLResponse)
    def export_page() -> HTMLResponse:
        """Store methods: ``item_count(kind)``, ``completed_item_keys()``, ``disagreements()``."""
        with store_session() as store:
            summary = {
                "relevance_items": store.item_count(KIND_RELEVANCE),
                "claim_items": store.item_count(KIND_CLAIM),
                "both_slots_complete": len(store.completed_item_keys()),
                "disagreements": len(store.disagreements()),
                "unadjudicated": len(store.disagreements(unadjudicated_only=True)),
            }
        return page("export.html", title="Export human labels", nav=adjudicator_nav(),
                    script="/static/export.js", summary=summary,
                    export_dir=str(exports), release_dir=str(releases))

    @app.post("/api/export")
    def run_export() -> dict:
        """Store method: ``export.export_annotations(store, ...)`` (analysis side).

        The destination directories are fixed at app creation, so this endpoint accepts no
        path from the request.
        """
        with store_session() as store:
            result = export_annotations(store, exports, release_dir=releases)
        return {
            "relevance_csv": str(result.relevance_path),
            "claims_csv": str(result.claims_path),
            "dump": str(result.dump_path),
            "manifest": str(result.manifest_path),
            "row_counts": result.row_counts,
            "sha256": result.hashes,
            "incomplete": {
                KIND_RELEVANCE: result.incomplete_count(KIND_RELEVANCE),
                KIND_CLAIM: result.incomplete_count(KIND_CLAIM),
            },
            "counts": result.manifest["counts"],
            "export_id": result.manifest["export_id"],
        }

    # ------------------------------------------------------------------ error mapping
    # Store-level refusals become 4xx, never a 500: an unassigned item is a client asking for
    # something that is not theirs, and an out-of-range label is a bad request.
    @app.exception_handler(NotAssignedError)
    def _not_assigned(_request: Request, exc: NotAssignedError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(InvalidLabelError)
    def _invalid_label(_request: Request, exc: InvalidLabelError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(UnknownRaterError)
    def _unknown_rater(_request: Request, exc: UnknownRaterError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


def _check_kind(kind: str | None, *, required: bool = False) -> None:
    if kind is None and not required:
        return
    if kind not in KINDS:
        raise HTTPException(status_code=404,
                            detail=f"unknown item kind {kind!r}; expected one of {KINDS}")


def _resolve_item(store: AnnotationStore, rater_id: str, kind: str, item_key: str | None,
                  after: str | None):
    """Which item to show: an explicit one, the next pending one, or the one after a skip.

    Skipping moves FORWARD in this rater's recorded presentation order and wraps to the first
    still-pending item, so a skipped item comes back later in the pass instead of being lost.
    Every lookup is rater-scoped, so ``after``/``item`` naming somebody else's item raises
    ``NotAssignedError``.
    """
    if item_key:
        return store.rater_item(rater_id, item_key)
    pending = store.queue(rater_id, include_done=False, kind=kind)
    if after:
        current = store.rater_item(rater_id, after)
        later = [entry for entry in pending if entry.position > current.position]
        candidates = later or [entry for entry in pending if entry.item_key != after]
        return candidates[0] if candidates else None
    return pending[0] if pending else None


#: Environment variables read by :func:`app_factory`, which is what the printed uvicorn command
#: uses: ``create_app`` needs arguments, and ``uvicorn --factory`` can only call a zero-argument
#: callable.
ENV_ANNOTATION_DIR = "CMJCC_ANNOTATION_DIR"
ENV_EXPORT_DIR = "CMJCC_ANNOTATION_EXPORT_DIR"
ENV_RELEASE_DIR = "CMJCC_ANNOTATION_RELEASE_DIR"
ENV_SECRET_KEY = "CMJCC_ANNOTATION_SECRET"


def app_factory() -> FastAPI:
    """Build the app from environment variables, for ``uvicorn ... --factory``.

    Print the exact command with
    ``python -m jobrec_eval.annotation_ui serve --annotation-dir <dir>``.

    Raises:
        RuntimeError: :data:`ENV_ANNOTATION_DIR` is not set. Guessing a default would risk
            serving the wrong annotation session.
    """
    import os

    annotation_dir = os.environ.get(ENV_ANNOTATION_DIR)
    if not annotation_dir:
        raise RuntimeError(
            f"set {ENV_ANNOTATION_DIR} to the directory holding {DB_FILENAME} before starting "
            f"the annotation UI")
    return create_app(annotation_dir,
                      secret_key=os.environ.get(ENV_SECRET_KEY),
                      export_dir=os.environ.get(ENV_EXPORT_DIR),
                      release_dir=os.environ.get(ENV_RELEASE_DIR))


#: Re-exported so a caller does not need to import :mod:`views` to know the flag string.
UNRESOLVABLE_EVIDENCE_FLAG = FLAG_UNRESOLVABLE_EVIDENCE
