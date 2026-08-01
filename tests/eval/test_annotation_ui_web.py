"""The annotation web UI, driven with TestClient over a store built from a REAL experiment.

Everything here goes through HTTP, because the properties that matter are properties of the
served app rather than of the data layer (which ``test_annotation_ui_store.py`` already covers):

- a rater's identity comes from the signed session cookie, and switching identity works;
- the relevance and grounding screens really render the payload fields
  :mod:`jobrec_eval.annotation_ui.loader` builds -- conversation turns in order, posting field
  values, resolved evidence, and the unresolvable-citation alert;
- a saved label reaches the store with its client-measured ``duration_ms``;
- **blinding**: no rater-facing response carries a blinded field name or the other rater's
  label, and no rater route so much as calls the analysis-side store methods (asserted by
  making those methods explode);
- **rater isolation**: another rater's item is a 404 over HTTP, for reads and for writes, and a
  ``rater_id`` in a request body cannot redirect a write;
- an invalid label is a 4xx, not a 500;
- the adjudicator's route records a verdict and is linked from no rater page;
- the export route writes the CSVs the pipeline consumes.

Rater ids are ``SYNTHETIC-WEB-RATER-*`` and every label posted here is invented by the test.
Nothing in this file may be mistaken for a collected human judgement.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from jobrec_eval.annotation_ui import console
from jobrec_eval.annotation_ui.app import RATER_COOKIE, create_app, is_loopback_host, sign_cookie
from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.export import CLAIMS_CSV_FILENAME, RELEVANCE_CSV_FILENAME
from jobrec_eval.annotation_ui.loader import build_items
from jobrec_eval.annotation_ui.store import (
    BLINDED_FIELD_NAMES,
    KIND_CLAIM,
    KIND_RELEVANCE,
    AnnotationItem,
    AnnotationStore,
    open_store,
)
from jobrec_eval.annotation_ui.templating import TemplateError, TemplateRenderer, parse
from jobrec_eval.annotation_ui.views import FLAG_UNRESOLVABLE_EVIDENCE

#: Three raters, not two: with a pool of exactly two every item lands on both queues and there
#: would be no item to prove the isolation boundary with.
RATER_POOL = ["SYNTHETIC-WEB-RATER-01", "SYNTHETIC-WEB-RATER-02", "SYNTHETIC-WEB-RATER-03"]
SECRET = "SYNTHETIC-TEST-SECRET"
SEED = 2026

#: Rater-facing HTML routes, checked as a set by the blinding and no-link tests.
RATER_HTML_ROUTES = ("/rater", "/queue", "/progress",
                     f"/annotate/{KIND_RELEVANCE}", f"/annotate/{KIND_CLAIM}")

#: Rater-facing JSON routes.
RATER_JSON_ROUTES = ("/api/queue", "/api/progress")


@dataclass
class Web:
    """A real-item annotation store plus the app serving it."""

    store: AnnotationStore
    app: Any
    directory: Path
    export_dir: Path
    release_dir: Path

    def client(self, rater_id: str | None = None) -> TestClient:
        """A client, optionally with the identity cookie already set for ``rater_id``."""
        client = TestClient(self.app)
        if rater_id is not None:
            response = client.post("/api/session", json={"rater_id": rater_id})
            assert response.status_code == 200, response.text
        return client

    def keys_for(self, rater_id: str, kind: str | None = None) -> list[str]:
        return [item.item_key for item in self.store.queue(rater_id, kind=kind)]

    def foreign_key(self, rater_id: str, kind: str | None = None) -> str:
        """An item that exists but is NOT assigned to ``rater_id``."""
        mine = set(self.keys_for(rater_id))
        for other in RATER_POOL:
            if other == rater_id:
                continue
            for key in self.keys_for(other, kind=kind):
                if key not in mine:
                    return key
        raise AssertionError("no item is assigned away from this rater; pool too small")

    def shared_key(self, first: str, second: str, kind: str | None = None) -> str:
        """An item assigned to BOTH raters, i.e. one they will be compared on."""
        other = set(self.keys_for(second, kind=kind))
        for key in self.keys_for(first, kind=kind):
            if key in other:
                return key
        raise AssertionError("the two raters share no item")


@pytest.fixture(scope="module")
def web(annotation_experiment, tmp_path_factory) -> Web:
    """Build items from the real experiment, assign three raters, serve the app over it."""
    root = tmp_path_factory.mktemp("annotation-web")
    directory = root / "annotation"
    export_dir = root / "csv"
    release_dir = root / "human_annotations"
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path)
    store = open_store(directory)
    store.register_raters(RATER_POOL)
    store.add_items(built.all_items)
    store.save_assignment_plan(assign_two_raters(store.item_keys(), RATER_POOL, SEED))
    app = create_app(directory, secret_key=SECRET, export_dir=export_dir,
                     release_dir=release_dir)
    try:
        yield Web(store=store, app=app, directory=directory, export_dir=export_dir,
                  release_dir=release_dir)
    finally:
        store.close()


def _shows(text: str, value: Any) -> bool:
    """Is ``value`` rendered in ``text``, allowing for HTML escaping?"""
    rendered = str(value)
    return rendered in text or html.escape(rendered, quote=True) in text


def _assert_shows(text: str, value: Any, what: str) -> None:
    assert _shows(text, value), f"{what} is missing from the rendered screen: {value!r}"


# --------------------------------------------------------------------- rater selection
def test_root_redirects_to_rater_selection_and_lists_the_registered_pool(web):
    client = TestClient(web.app)
    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/rater"

    page = client.get("/rater")
    assert page.status_code == 200
    for rater in RATER_POOL:
        _assert_shows(page.text, rater, "registered rater")
    # The screen must say what the choice is and is not.
    assert "not a login" in page.text.lower()


def test_selecting_a_rater_sets_a_signed_http_only_cookie_and_switching_clears_it(web):
    client = TestClient(web.app)
    response = client.post("/api/session", json={"rater_id": RATER_POOL[0]})
    assert response.status_code == 200
    assert response.json() == {"rater_id": RATER_POOL[0], "next_url": "/queue"}
    cookie_header = response.headers["set-cookie"]
    assert RATER_COOKIE in cookie_header
    assert "httponly" in cookie_header.lower()
    assert "samesite=lax" in cookie_header.lower()
    # The cookie carries a signature, not a bare rater id.
    assert client.cookies[RATER_COOKIE].split(".")[0] != RATER_POOL[0]

    queue = client.get("/queue")
    assert queue.status_code == 200
    _assert_shows(queue.text, RATER_POOL[0], "the selected rater")

    switched = client.post("/api/session/clear")
    assert switched.status_code == 200
    assert client.get("/queue", follow_redirects=False).headers["location"] == "/rater"

    # Switching to somebody else works, and an unregistered name is refused.
    assert client.post("/api/session", json={"rater_id": RATER_POOL[1]}).status_code == 200
    _assert_shows(client.get("/queue").text, RATER_POOL[1], "the newly selected rater")
    assert client.post("/api/session", json={"rater_id": "SYNTHETIC-NOT-REGISTERED"
                                             }).status_code == 400


def test_a_forged_or_tampered_identity_cookie_is_not_accepted(web):
    """The cookie is signed, so a hand-written one does not select a rater."""
    client = TestClient(web.app)
    client.cookies.set(RATER_COOKIE, f"{RATER_POOL[0]}.not-a-signature")
    assert client.get("/queue", follow_redirects=False).headers["location"] == "/rater"

    # A cookie signed with a different key is refused too.
    client.cookies.set(RATER_COOKIE, sign_cookie({"rater_id": RATER_POOL[0]}, "WRONG-KEY"))
    assert client.get("/queue", follow_redirects=False).headers["location"] == "/rater"

    client.cookies.set(RATER_COOKIE, sign_cookie({"rater_id": RATER_POOL[0]}, SECRET))
    assert client.get("/queue").status_code == 200


# ------------------------------------------------------------------ annotation screens
def test_the_relevance_screen_renders_the_real_payload_fields(web):
    rater = RATER_POOL[0]
    client = web.client(rater)
    key = web.keys_for(rater, KIND_RELEVANCE)[0]
    item = web.store.rater_item(rater, key)
    payload = item.payload
    scenario, job = payload["scenario"], payload["job"]

    page = client.get(f"/annotate/{KIND_RELEVANCE}", params={"item": key})
    assert page.status_code == 200
    text = page.text

    _assert_shows(text, payload["task"], "the task instruction")
    _assert_shows(text, scenario["scenario_id"], "scenario id")
    for value in scenario["candidate_profile"].values():
        if isinstance(value, str) and value:
            _assert_shows(text, value, "candidate profile value")

    # Conversation turns are all present AND in order: a later turn can revise an earlier
    # preference, so the sequence is part of the evidence a rater judges on.
    turns = scenario["conversation"]
    assert turns, "the fixture scenario has no conversation turns to render"
    positions = []
    for turn in turns:
        utterance = turn["candidate_utterance"]
        _assert_shows(text, utterance, "candidate utterance")
        needle = utterance if utterance in text else html.escape(utterance, quote=True)
        positions.append(text.index(needle))
    assert positions == sorted(positions), "conversation turns are not rendered in order"

    # The posting side.
    for field in ("job_id", "title", "company", "work_mode", "employment_type",
                  "experience_level"):
        if job.get(field):
            _assert_shows(text, job[field], f"job {field}")
    _assert_shows(text, job["description"], "job description")
    for skill in job["required_skills"]:
        _assert_shows(text, skill, "required skill")
    for responsibility in job["responsibilities"]:
        _assert_shows(text, responsibility, "responsibility")
    if job["location"]["city"]:
        _assert_shows(text, job["location"]["city"], "location city")

    # The rubric is visible, not hidden behind a tooltip.
    for grade, description in payload["grade_scale"].items():
        _assert_shows(text, description, f"rubric wording for grade {grade}")
        assert f'data-label="{grade}"' in text, f"no button for grade {grade}"

    # Notes field is a real labelled control, and the shortcuts are documented on screen.
    assert '<label for="notes"' in text
    assert 'id="notes"' in text
    assert "Keyboard shortcuts" in text
    assert 'id="save"' in text and 'id="skip"' in text


def test_the_grounding_screen_renders_the_claim_its_evidence_and_the_posting(web):
    rater = RATER_POOL[0]
    client = web.client(rater)
    key = web.keys_for(rater, KIND_CLAIM)[0]
    payload = web.store.rater_item(rater, key).payload

    page = client.get(f"/annotate/{KIND_CLAIM}", params={"item": key})
    assert page.status_code == 200
    text = page.text

    _assert_shows(text, payload["claim_text"], "the claim sentence")
    _assert_shows(text, payload["claim_id"], "claim id")
    _assert_shows(text, payload["claim_type"], "claim type")
    _assert_shows(text, payload["cited_evidence_count"], "cited evidence count")
    _assert_shows(text, payload["occurrence_count"], "occurrence count")

    assert payload["evidence"], "the fixture claim resolved no evidence to render"
    for record in payload["evidence"]:
        _assert_shows(text, record["field_name"], "evidence field name")
        _assert_shows(text, record["source"], "evidence source")
        if record.get("source_object_id"):
            _assert_shows(text, record["source_object_id"], "evidence source object id")
        if record.get("raw_text"):
            _assert_shows(text, record["raw_text"], "evidence raw text")

    for job in payload["referenced_jobs"]:
        _assert_shows(text, job["job_id"], "referenced job id")
        if job.get("title"):
            _assert_shows(text, job["title"], "referenced job title")

    # Both labels are offered as buttons with words, not colours.
    for label, description in payload["label_scale"].items():
        assert f'data-label="{label}"' in text
        _assert_shows(text, description, f"wording for label {label}")

    # The unresolvable-evidence flag is always available, keyboard shortcut documented.
    assert f'data-flag-value="{FLAG_UNRESOLVABLE_EVIDENCE}"' in text
    assert 'id="flag-unresolvable"' in text
    assert '<label for="flag-unresolvable"' in text
    assert "<kbd>E</kbd>" in text


def test_an_unresolvable_citation_is_shown_prominently_and_can_be_flagged(annotation_experiment,
                                                                         tmp_path_factory):
    """A dangling citation must be impossible to miss, and the flag must reach the store.

    Built as its own store from a REAL claim payload with the unresolvable list populated: the
    checklist asks a rater to check whether a claim's evidence ids resolve, so the screen has to
    say so loudly rather than leaving the rater to notice a missing row.
    """
    root = tmp_path_factory.mktemp("annotation-web-unresolvable")
    directory = root / "annotation"
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path)
    source = built.claim_items[0]
    payload = dict(source.payload)
    payload["unresolvable_evidence_ids"] = ["SYNTHETIC-EV-DOES-NOT-RESOLVE"]
    payload["has_unresolvable_evidence"] = True
    item = AnnotationItem(item_key="clm::sig-SYNTHETICUNRESOLV", kind=KIND_CLAIM,
                          annotation_signature="sig-SYNTHETICUNRESOLV",
                          payload=payload, analysis={}, claim_id="SYNTHETIC-UNRESOLVABLE")
    pool = RATER_POOL[:2]
    with open_store(directory) as store:
        store.register_raters(pool)
        store.add_items([item])
        store.save_assignment_plan(assign_two_raters(store.item_keys(), pool, SEED))

        app = create_app(directory, secret_key=SECRET)
        client = TestClient(app)
        assert client.post("/api/session", json={"rater_id": pool[0]}).status_code == 200
        page = client.get(f"/annotate/{KIND_CLAIM}", params={"item": item.item_key})
        assert page.status_code == 200
        text = page.text

        assert 'role="alert"' in text
        _assert_shows(text, "SYNTHETIC-EV-DOES-NOT-RESOLVE", "the unresolvable evidence id")
        assert "cannot support anything" in text
        # The alert comes before the decision controls, so it cannot be scrolled past unseen.
        assert text.index("SYNTHETIC-EV-DOES-NOT-RESOLVE") < text.index('id="save"')

        saved = client.post("/api/annotations", json={
            "item_key": item.item_key, "label": 0, "notes": "citation does not resolve",
            "flags": FLAG_UNRESOLVABLE_EVIDENCE, "duration_ms": 9100})
        assert saved.status_code == 200
        record = store.annotation(item.item_key, pool[0])
        assert record is not None
        assert record.flags == FLAG_UNRESOLVABLE_EVIDENCE
        assert record.label == 0


def test_saving_a_label_persists_through_the_store_with_the_measured_duration(web):
    rater = RATER_POOL[1]
    client = web.client(rater)
    key = web.keys_for(rater, KIND_RELEVANCE)[0]

    response = client.post("/api/annotations", json={
        "item_key": key, "label": 2, "notes": "SYNTHETIC note", "duration_ms": 7345})
    assert response.status_code == 200
    body = response.json()
    assert body["saved"] is True
    assert body["label"] == 2
    assert body["duration_ms"] == 7345
    assert body["revised"] is False

    record = web.store.annotation(key, rater)
    assert record is not None
    assert (record.label, record.notes, record.duration_ms) == (2, "SYNTHETIC note", 7345)
    assert web.store.progress(rater).median_duration_ms is not None

    # Saving again revises the rater's OWN answer and keeps the first-save timestamp.
    revised = client.post("/api/annotations", json={
        "item_key": key, "label": 3, "notes": "SYNTHETIC revised", "duration_ms": 2100})
    assert revised.status_code == 200
    assert revised.json()["revised"] is True
    assert web.store.annotation(key, rater).label == 3

    # next_url points at another item this rater still owes, or back to the queue.
    next_url = body["next_url"]
    assert next_url == "/queue" or next_url.startswith(f"/annotate/{KIND_RELEVANCE}?item=")


def test_next_and_skip_walk_the_queue_without_writing_anything(web):
    rater = RATER_POOL[2]
    client = web.client(rater)
    pending = [item.item_key for item in web.store.queue(rater, include_done=False,
                                                         kind=KIND_RELEVANCE)]
    assert len(pending) >= 2, "need at least two pending items to prove skip advances"

    first = client.get(f"/annotate/{KIND_RELEVANCE}")
    assert first.status_code == 200
    assert f'data-item-key="{pending[0]}"' in first.text

    skipped = client.get(f"/annotate/{KIND_RELEVANCE}", params={"after": pending[0]})
    assert skipped.status_code == 200
    assert f'data-item-key="{pending[0]}"' not in skipped.text
    # Skipping wrote nothing: the item is still unlabelled and still on the queue.
    assert web.store.annotation(pending[0], rater) is None
    assert pending[0] in [item.item_key for item in
                          web.store.queue(rater, include_done=False, kind=KIND_RELEVANCE)]


# --------------------------------------------------------------------------- blinding
def test_no_rater_facing_response_carries_a_blinded_field_or_the_other_raters_label(web):
    """Blinding and independence, asserted on the actual response bodies.

    ``BLINDED_FIELD_NAMES`` is imported rather than spelled out, so adding a blinded name to the
    store automatically tightens this test.
    """
    rater, other = RATER_POOL[0], RATER_POOL[1]
    shared = web.shared_key(rater, other, kind=KIND_RELEVANCE)
    # The other rater answers first, through their own session, so there IS a label to leak.
    other_client = web.client(other)
    assert other_client.post("/api/annotations", json={
        "item_key": shared, "label": 3, "notes": "SYNTHETIC other rater note",
        "duration_ms": 5000}).status_code == 200

    client = web.client(rater)
    bodies: dict[str, str] = {}
    for route in RATER_HTML_ROUTES + RATER_JSON_ROUTES:
        response = client.get(route)
        assert response.status_code == 200, f"{route} -> {response.status_code}"
        bodies[route] = response.text
    bodies["/annotate/item"] = client.get(f"/annotate/{KIND_RELEVANCE}",
                                          params={"item": shared}).text
    bodies["/api/item"] = client.get("/api/item", params={"item": shared}).text
    bodies["/api/annotations"] = client.post("/api/annotations", json={
        "item_key": shared, "label": 1, "notes": "SYNTHETIC own note",
        "duration_ms": 4000}).text

    for route, body in bodies.items():
        for blinded in sorted(BLINDED_FIELD_NAMES):
            assert blinded not in body, f"{route} leaked blinded field name {blinded!r}"
        # The selection screen lists the pool on purpose -- that is how you pick your own name.
        # Everywhere else, another rater's id has no business appearing: it could only be there
        # to tie them to an item, and their answer is what must stay unseen.
        if route != "/rater":
            for stranger in RATER_POOL:
                if stranger != rater:
                    assert stranger not in body, f"{route} named another rater ({stranger})"
        for adjudication_shape in ("slot_1_label", "slot_2_label", "rater_1", "rater_2",
                                   "adjudicated", "disagreement"):
            assert adjudication_shape not in body, (
                f"{route} carries adjudication-shaped data ({adjudication_shape})")
        # And no route to the screens that DO show that data.
        assert "/adjudication" not in body, f"{route} links to the adjudication queue"
        assert "/api/export" not in body, f"{route} links to the export endpoint"

    # The other rater's own answer is untouched by everything above.
    assert web.store.annotation(shared, other).label == 3
    assert web.store.annotation(shared, rater).label == 1


def test_rater_routes_never_call_the_analysis_side_store_methods(web, monkeypatch):
    """Mechanical version of the blinding boundary: make the analysis methods explode.

    ``disagreements`` and ``iter_export_records`` are the only store methods that can return the
    other rater's label, the oracle grade or the validator verdict. If a rater route grew a call
    to either -- even indirectly -- these requests would fail instead of quietly unblinding
    somebody.
    """
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a rater route called an analysis-side store method")

    monkeypatch.setattr(AnnotationStore, "disagreements", forbidden)
    monkeypatch.setattr(AnnotationStore, "iter_export_records", forbidden)

    rater = RATER_POOL[0]
    client = web.client(rater)
    key = web.keys_for(rater, KIND_CLAIM)[0]
    for route, params in (("/rater", None), ("/queue", None), ("/progress", None),
                          (f"/annotate/{KIND_RELEVANCE}", None),
                          (f"/annotate/{KIND_CLAIM}", {"item": key}),
                          ("/api/queue", None), ("/api/progress", None),
                          ("/api/item", {"item": key})):
        response = client.get(route, params=params)
        assert response.status_code == 200, f"{route} -> {response.status_code} {response.text}"
    assert client.post("/api/annotations", json={
        "item_key": key, "label": 1, "duration_ms": 3000}).status_code == 200


# --------------------------------------------------------------------- rater isolation
def test_a_rater_cannot_read_another_raters_item_over_http(web):
    rater = RATER_POOL[0]
    client = web.client(rater)
    foreign = web.foreign_key(rater)

    kind = KIND_RELEVANCE if foreign.startswith("rel::") else KIND_CLAIM
    page = client.get(f"/annotate/{kind}", params={"item": foreign})
    assert page.status_code == 404
    assert "not on your queue" in page.text.lower()
    # The refusal page must not smuggle the item's content in as consolation.
    assert "candidate-heading" not in page.text

    assert client.get("/api/item", params={"item": foreign}).status_code == 404
    # Skipping "past" somebody else's item is refused the same way.
    assert client.get(f"/annotate/{kind}", params={"after": foreign}).status_code == 404


def test_a_rater_cannot_write_another_raters_item_and_cannot_nominate_a_rater_id(web):
    rater, victim = RATER_POOL[0], RATER_POOL[1]
    foreign = web.foreign_key(rater)
    client = web.client(rater)

    # Snapshot first: the assertion is that the REFUSED post changed nothing, not that the
    # item has never been written. Its rightful owner may legitimately have annotated it in
    # an earlier test against this shared fixture, and asserting `is None` conflated the two
    # -- which made the test depend on where this item happened to fall in the generated
    # order, so an unrelated change to the item set could break it.
    owners = [owner for owner in RATER_POOL if foreign in web.keys_for(owner)]
    before = {owner: web.store.annotation(foreign, owner) for owner in owners}

    refused = client.post("/api/annotations", json={"item_key": foreign, "label": 1,
                                                   "duration_ms": 1000})
    assert refused.status_code == 404, refused.text
    for owner in owners:
        after = web.store.annotation(foreign, owner)
        assert (after.label if after else None) == (
            before[owner].label if before[owner] else None)
    # No separate check that the posting rater has no record: the store raises
    # NotAssignedError for an item that is not theirs, which is the same refusal the 404
    # above already established.
    assert rater not in owners

    # A rater_id in the body is ignored: the write lands on the SESSION rater, always.
    mine = web.shared_key(rater, victim, kind=KIND_CLAIM)
    before = web.store.annotation(mine, victim)
    response = client.post("/api/annotations", json={
        "item_key": mine, "label": 0, "notes": "SYNTHETIC session-owned",
        "duration_ms": 2500, "rater_id": victim})
    assert response.status_code == 200
    assert web.store.annotation(mine, rater).notes == "SYNTHETIC session-owned"
    after = web.store.annotation(mine, victim)
    assert (after.label if after else None) == (before.label if before else None)


def test_writing_without_choosing_a_rater_is_refused(web):
    client = TestClient(web.app)
    key = web.keys_for(RATER_POOL[0])[0]
    assert client.post("/api/annotations", json={"item_key": key, "label": 1}).status_code == 403
    assert client.get("/api/progress").status_code == 403
    assert client.get("/api/item", params={"item": key}).status_code == 403


def test_an_invalid_label_is_a_4xx_not_a_500(web):
    rater = RATER_POOL[0]
    client = web.client(rater)
    key = web.keys_for(rater, KIND_RELEVANCE)[0]
    claim_key = web.keys_for(rater, KIND_CLAIM)[0]

    for body, expected in (
        ({"item_key": key, "label": 4}, 422),                     # outside 0-3
        ({"item_key": key, "label": -1}, 422),
        ({"item_key": claim_key, "label": 2}, 422),               # claims are {0, 1}
        ({"item_key": key, "label": "three"}, 422),               # not an integer
        ({"item_key": key}, 422),                                 # no label at all
        ({"item_key": key, "label": 1, "duration_ms": -5}, 422),  # impossible duration
        ({"item_key": "", "label": 1}, 422),
    ):
        response = client.post("/api/annotations", json=body)
        assert response.status_code == expected, f"{body} -> {response.status_code}"
        assert response.status_code < 500

    assert client.get("/annotate/not-a-kind").status_code == 404


# ------------------------------------------------------------------------ adjudication
def test_the_adjudication_route_records_a_verdict_and_is_reachable_only_on_its_own(web):
    first, second = RATER_POOL[0], RATER_POOL[1]
    item_key = web.shared_key(first, second, kind=KIND_RELEVANCE)
    # Two raters, two different labels, each through their own session.
    assert web.client(first).post("/api/annotations", json={
        "item_key": item_key, "label": 0, "duration_ms": 3000}).status_code == 200
    assert web.client(second).post("/api/annotations", json={
        "item_key": item_key, "label": 3, "duration_ms": 3200}).status_code == 200

    adjudicator = TestClient(web.app)
    page = adjudicator.get("/adjudication")
    assert page.status_code == 200
    # Both raters' labels are side by side here -- and only here.
    _assert_shows(page.text, item_key, "the disagreeing item")
    _assert_shows(page.text, first, "slot 1 rater")
    _assert_shows(page.text, second, "slot 2 rater")
    assert 'name="adjudicator"' in page.text

    queue = adjudicator.get("/api/adjudication/queue", params={"open_only": "1"}).json()
    assert queue["count"] >= 1
    case = next(entry for entry in queue["cases"] if entry["item_key"] == item_key)
    assert {case["slot_1_label"], case["slot_2_label"]} == {0, 3}

    # The reason is required; an unnamed adjudicator is refused too.
    assert adjudicator.post("/api/adjudication/verdicts", json={
        "item_key": item_key, "final_label": 2, "reason": "  ",
        "adjudicator": "SYNTHETIC-ADJUDICATOR"}).status_code == 422
    assert adjudicator.post("/api/adjudication/verdicts", json={
        "item_key": item_key, "final_label": 2, "reason": "resolved"}).status_code == 422
    assert adjudicator.post("/api/adjudication/verdicts", json={
        "item_key": item_key, "final_label": 9, "reason": "out of range",
        "adjudicator": "SYNTHETIC-ADJUDICATOR"}).status_code == 422

    recorded = adjudicator.post("/api/adjudication/verdicts", json={
        "item_key": item_key, "final_label": 2, "reason": "SYNTHETIC adjudication reason",
        "adjudicator": "SYNTHETIC-ADJUDICATOR"})
    assert recorded.status_code == 200
    verdict = web.store.adjudication(item_key)
    assert verdict is not None
    assert (verdict.final_label, verdict.adjudicator) == (2, "SYNTHETIC-ADJUDICATOR")
    assert verdict.reason == "SYNTHETIC adjudication reason"

    # No rater screen offers a way in.
    rater_client = web.client(RATER_POOL[2])
    for route in RATER_HTML_ROUTES:
        assert "/adjudication" not in rater_client.get(route).text, route


# ------------------------------------------------------------------------------ export
def test_the_export_route_writes_the_csvs_the_pipeline_consumes(web):
    adjudicator = TestClient(web.app)
    page = adjudicator.get("/export")
    assert page.status_code == 200
    _assert_shows(page.text, str(web.export_dir), "the configured export directory")

    response = adjudicator.post("/api/export")
    assert response.status_code == 200
    body = response.json()

    relevance_path = Path(body["relevance_csv"])
    claims_path = Path(body["claims_csv"])
    assert relevance_path.name == RELEVANCE_CSV_FILENAME
    assert claims_path.name == CLAIMS_CSV_FILENAME
    assert relevance_path.is_file() and claims_path.is_file()
    assert Path(body["manifest"]).is_file() and Path(body["dump"]).is_file()

    relevance = pd.read_csv(relevance_path)
    claims = pd.read_csv(claims_path)
    assert len(relevance) == body["row_counts"][RELEVANCE_CSV_FILENAME]
    assert len(claims) == body["row_counts"][CLAIMS_CSV_FILENAME]
    assert set(body["sha256"]) == {RELEVANCE_CSV_FILENAME, CLAIMS_CSV_FILENAME}
    assert all(len(digest) == 64 for digest in body["sha256"].values())
    # Items only one rater has answered are counted, not written as blank rows.
    assert body["incomplete"][KIND_RELEVANCE] >= 0
    assert body["counts"]["relevance_items"] == web.store.item_count(KIND_RELEVANCE)
    if not relevance.empty:
        assert list(relevance.columns)[:4] == ["scenario_id", "job_id", "rater_1", "rater_2"]
        assert relevance["rater_1"].between(0, 3).all()


# -------------------------------------------------------------------- static behaviour
def test_the_served_javascript_carries_the_documented_keyboard_bindings(web):
    client = TestClient(web.app)
    response = client.get("/static/annotate.js")
    assert response.status_code == 200
    script = response.text

    assert "KEY_BINDINGS" in script
    assert "relevanceGrades: ['0', '1', '2', '3']" in script
    assert "claimLabels: ['1', '0']" in script
    assert "saveAndNext: ['Enter', 'n', 'N']" in script
    assert "skip: ['s', 'S']" in script
    assert "flagUnresolvableEvidence: ['e', 'E']" in script
    # The client-side timer is what produces duration_ms.
    assert "duration_ms" in script
    assert "performance.now" in script
    # Shortcuts must not fire while the rater is typing a note.
    assert "typingInField" in script

    assert client.get("/static/app.css").status_code == 200
    assert ":focus-visible" in client.get("/static/app.css").text


def test_every_page_is_keyboard_reachable_and_announces_status(web):
    """Accessibility scaffolding that is checkable without a screen reader."""
    client = web.client(RATER_POOL[0])
    for route in RATER_HTML_ROUTES:
        text = client.get(route).text
        assert 'class="skip-link"' in text, route
        assert 'id="main"' in text, route
        assert 'aria-live="polite"' in text, route
        assert 'role="status"' in text, route
        assert "<html lang=\"en\">" in text, route


# ---------------------------------------------------------------------- serve command
def test_serve_prints_the_url_and_the_uvicorn_command_without_starting_a_server(web, capsys):
    assert console.main(["serve", "--annotation-dir", str(web.directory)]) == 0
    printed = json.loads(capsys.readouterr().out)

    assert printed["started"] is False
    assert printed["url"] == "http://127.0.0.1:8765/"
    assert printed["loopback"] is True
    assert printed["authentication"] == "none"
    assert "uvicorn" in printed["uvicorn_command"]
    assert "--factory" in printed["uvicorn_command"]
    assert printed["environment"]["CMJCC_ANNOTATION_DIR"] == str(web.directory)
    assert any("no login" in warning.lower() or "authentication" in warning.lower()
               for warning in printed["warnings"])


def test_serve_refuses_a_non_loopback_bind_without_an_explicit_opt_in(web, capsys):
    with pytest.raises(SystemExit) as refused:
        console.main(["serve", "--annotation-dir", str(web.directory), "--host", "0.0.0.0"])
    assert "authentication" in str(refused.value).lower()

    assert console.main(["serve", "--annotation-dir", str(web.directory),
                         "--host", "0.0.0.0", "--allow-remote-host"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["loopback"] is False
    assert any("NOT A LOOPBACK" in warning for warning in printed["warnings"])

    assert is_loopback_host("127.0.0.1") and is_loopback_host("localhost")
    assert is_loopback_host("::1") and not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("") and not is_loopback_host("example.com")


def test_the_app_refuses_to_serve_a_directory_with_no_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "nothing-here")


# ------------------------------------------------------------------ template renderer
def test_the_template_renderer_escapes_and_supports_the_subset_the_templates_use(tmp_path):
    """Jinja2 is absent, so the renderer's own guarantees are worth asserting.

    Escaping matters for real data: a job description or a candidate utterance containing ``<``
    must render as text on a rater's screen, not as markup.
    """
    (tmp_path / "base.html").write_text("<main>{{ content | raw }}</main>", encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "{% if rows %}<ul>{% for row in rows %}<li>{{ row.name }}:"
        "{% for tag in row.tags %}<em>{{ tag }}</em>{% endfor %}</li>{% endfor %}</ul>"
        "{% else %}<p>none</p>{% endif %}{% if not missing %}<p>{{ absent }}</p>{% endif %}",
        encoding="utf-8")
    renderer = TemplateRenderer(tmp_path)

    rendered = renderer.render_page("page.html", {
        "rows": [{"name": "<script>alert(1)</script>", "tags": ["a & b"]}]})
    assert rendered.startswith("<main>") and rendered.endswith("</main>")
    assert "<script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<em>a &amp; b</em>" in rendered
    assert "<p></p>" in rendered  # a missing value renders empty, it does not raise

    assert renderer.render("page.html", {"rows": []}) == "<p>none</p><p></p>"

    with pytest.raises(TemplateError):
        parse("{% while true %}{% endwhile %}")
    with pytest.raises(TemplateError):
        parse("{% for row in rows %}no end tag")
    with pytest.raises(TemplateError):
        parse("{{ 1 + 1 }}")
