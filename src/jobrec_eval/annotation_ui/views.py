"""View models: one rater-facing payload -> plain dicts and lists a template can render.

Everything a screen shows is built here, from the payload keys
:mod:`~jobrec_eval.annotation_ui.loader` writes and from NOTHING else. Two consequences that
matter more than tidiness:

- the functions take a :class:`~jobrec_eval.annotation_ui.store.RaterItem`, which has no field
  for the oracle grade or the validator verdict, so there is no value in scope here that could
  leak onto a rater's screen (blinding, checklist items 10/11);
- the templates need no expression language, because every list, label and formatted value is
  computed in Python where it is unit-testable.

Presentation decisions that are substantive rather than cosmetic:

- conversation turns are sorted by ``turn_index`` and rendered in sequence, because a later
  turn can revise an earlier preference and a rater judging fit has to read them in order;
- ``unresolvable_evidence_ids`` becomes its own alert block at the TOP of a claim screen, plus
  a pre-wired flag. A citation that resolves to nothing cannot support anything, so hiding it
  in a list of evidence rows would invite a "supported" label for a claim whose citation is
  dangling -- exactly the case checklist item 11 asks a rater to check;
- a missing job or scenario is shown as an explicit "not in the snapshot" notice instead of an
  empty panel, so a rater knows the comparison is impossible rather than blank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .store import KIND_CLAIM, KIND_RELEVANCE, LABEL_RANGES, RaterItem

#: Flag string recorded on a claim annotation when a rater confirms a cited evidence id does
#: not resolve. Written through ``AnnotationStore.upsert_annotation(..., flags=...)`` and
#: carried into the archive dump, so the count is recoverable after the pass.
FLAG_UNRESOLVABLE_EVIDENCE = "evidence_id_does_not_resolve"

#: Placeholder for a field the source data does not carry. A visible dash beats an empty cell:
#: the rater can tell "not recorded" from "I missed it".
EMPTY = "not recorded"

#: ``(payload key, visible label)`` for the posting panel on a RELEVANCE screen, in reading
#: order. Long prose (``description``, ``responsibilities``) is rendered separately.
JOB_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("job_id", "Job id"),
    ("title", "Title"),
    ("company", "Company"),
    ("location", "Location"),
    ("work_mode", "Work mode"),
    ("employment_type", "Employment type"),
    ("salary", "Salary"),
    ("required_skills", "Required skills"),
    ("preferred_skills", "Preferred skills"),
    ("min_years_experience", "Minimum years experience"),
    ("experience_level", "Experience level"),
    ("application_deadline", "Application deadline"),
    ("is_active", "Posting active"),
)

#: Evidence-record fields a rater judges a citation by, in reading order.
EVIDENCE_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("field_name", "Field"),
    ("normalized_value", "Normalized value"),
    ("raw_text", "Raw text"),
    ("source", "Source"),
    ("source_object_id", "Source object id"),
)


def format_value(value: Any) -> str:
    """Human-readable text for any payload value, never a bare ``None`` or ``{}``."""
    if value is None or value == "":
        return EMPTY
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        parts = [f"{key}: {format_value(inner)}" for key, inner in value.items()
                 if inner not in (None, "")]
        return "; ".join(parts) if parts else EMPTY
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [format_value(inner) for inner in value if inner not in (None, "")]
        return ", ".join(parts) if parts else EMPTY
    return str(value)


def _format_location(location: Any) -> str:
    if not isinstance(location, Mapping):
        return format_value(location)
    parts = [str(location.get(key)) for key in ("city", "region", "country")
             if location.get(key)]
    return ", ".join(parts) if parts else EMPTY


def _format_salary(salary: Any) -> str:
    if not isinstance(salary, Mapping):
        return format_value(salary)
    low, high = salary.get("min"), salary.get("max")
    currency = salary.get("currency") or ""
    period = salary.get("period") or ""
    if low is None and high is None:
        return EMPTY
    if low is not None and high is not None:
        amount = f"{low} to {high}"
    else:
        amount = f"from {low}" if low is not None else f"up to {high}"
    return " ".join(part for part in (amount, currency, f"per {period}" if period else "") if part)


def _field_rows(payload: Mapping[str, Any],
                labels: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """``[{"label": ..., "value": ...}]`` for the fields present in a payload."""
    rows: list[dict[str, str]] = []
    for key, label in labels:
        if key not in payload:
            continue
        value = payload[key]
        if key == "location":
            text = _format_location(value)
        elif key == "salary":
            text = _format_salary(value)
        else:
            text = format_value(value)
        rows.append({"label": label, "value": text, "key": key})
    return rows


def _profile_rows(profile: Any) -> list[dict[str, str]]:
    if not isinstance(profile, Mapping):
        return []
    return [{"label": str(key).replace("_", " "), "value": format_value(value), "key": str(key)}
            for key, value in profile.items()]


def _turns(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Conversation turns in ``turn_index`` order, numbered from 1 for display."""
    raw = scenario.get("conversation") or []
    ordered = sorted(
        (turn for turn in raw if isinstance(turn, Mapping)),
        key=lambda turn: int(turn.get("turn_index") or 0))
    return [{"turn_index": int(turn.get("turn_index") or 0),
             "turn_number": position,
             "candidate_utterance": str(turn.get("candidate_utterance") or "")}
            for position, turn in enumerate(ordered, start=1)]


def _scale_options(scale: Any, kind: str) -> list[dict[str, str]]:
    """Grade/label buttons: ``{"value", "description", "hotkey"}``, highest value first.

    The valid values come from :data:`~jobrec_eval.annotation_ui.store.LABEL_RANGES` rather
    than from the payload's scale dict, so a screen can never offer a button the store would
    reject; the payload supplies the wording of the rubric.
    """
    descriptions = {str(key): str(value) for key, value in (scale or {}).items()} \
        if isinstance(scale, Mapping) else {}
    options: list[dict[str, str]] = []
    for value in sorted(LABEL_RANGES[kind], reverse=True):
        options.append({
            "value": str(value),
            "description": descriptions.get(str(value), ""),
            "hotkey": str(value),
        })
    return options


def _common(item: RaterItem) -> dict[str, Any]:
    return {
        "item_key": item.item_key,
        "kind": item.kind,
        "position": item.position,
        "slot": item.slot,
        "done": item.done,
        "label": "" if item.label is None else str(item.label),
        "notes": item.notes,
        "flags": item.flags,
        "task": str(item.payload.get("task") or ""),
    }


def relevance_view(item: RaterItem) -> dict[str, Any]:
    """View model for the relevance screen: candidate side, posting side, rubric."""
    payload = item.payload
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), Mapping) else {}
    job = payload.get("job") if isinstance(payload.get("job"), Mapping) else {}
    scenario = dict(scenario)
    job = dict(job)
    responsibilities = [str(entry) for entry in (job.get("responsibilities") or [])]
    slots = [str(entry) for entry in (scenario.get("acceptable_clarification_slots") or [])]
    view = {
        **_common(item),
        "scenario_id": str(scenario.get("scenario_id") or ""),
        "scenario_type": str(scenario.get("scenario_type") or EMPTY),
        "profile_rows": _profile_rows(scenario.get("candidate_profile")),
        "turns": _turns(scenario),
        "clarification_slots": ", ".join(slots) if slots else EMPTY,
        "scenario_missing": bool(scenario.get("missing_from_scenario_file")),
        "job_id": str(job.get("job_id") or ""),
        "job_title": str(job.get("title") or EMPTY),
        "job_company": str(job.get("company") or EMPTY),
        "job_rows": _field_rows(job, JOB_FIELD_LABELS),
        "job_description": str(job.get("description") or EMPTY),
        "job_responsibilities": responsibilities,
        "job_missing": bool(job.get("missing_from_catalog")),
        "grades": _scale_options(payload.get("grade_scale"), KIND_RELEVANCE),
    }
    view["has_turns"] = bool(view["turns"])
    view["has_profile"] = bool(view["profile_rows"])
    view["has_responsibilities"] = bool(responsibilities)
    return view


def claim_view(item: RaterItem) -> dict[str, Any]:
    """View model for the grounding screen: sentence, citations, posting fields, alert."""
    payload = item.payload
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), Mapping) else {}
    evidence = [entry for entry in (payload.get("evidence") or []) if isinstance(entry, Mapping)]
    unresolvable = [str(entry) for entry in (payload.get("unresolvable_evidence_ids") or [])]
    jobs = []
    for entry in payload.get("referenced_jobs") or []:
        if not isinstance(entry, Mapping):
            continue
        jobs.append({
            "job_id": str(entry.get("job_id") or ""),
            "title": str(entry.get("title") or EMPTY),
            "missing": bool(entry.get("missing_from_catalog")),
            # The claim payload carries the narrower ``loader.CLAIM_JOB_FIELDS`` subset;
            # ``_field_rows`` renders whichever of them are present and skips the rest.
            "rows": _field_rows(entry, JOB_FIELD_LABELS),
        })
    scenario_ids = [str(entry) for entry in (payload.get("scenario_ids") or [])]
    view = {
        **_common(item),
        "claim_id": str(payload.get("claim_id") or ""),
        "claim_type": str(payload.get("claim_type") or EMPTY),
        "claim_text": str(payload.get("claim_text") or ""),
        "cited_evidence_count": int(payload.get("cited_evidence_count") or 0),
        "occurrence_count": int(payload.get("occurrence_count") or 0),
        "evidence_merged_across_runs": bool(payload.get("evidence_merged_across_runs")),
        "evidence": [
            {"index": position,
             "rows": _field_rows(entry, EVIDENCE_FIELD_LABELS),
             "field_name": str(entry.get("field_name") or EMPTY),
             "normalized_value": format_value(entry.get("normalized_value")),
             "raw_text": str(entry.get("raw_text") or EMPTY),
             "source": str(entry.get("source") or EMPTY),
             "source_object_id": str(entry.get("source_object_id") or EMPTY)}
            for position, entry in enumerate(evidence, start=1)],
        "unresolvable_ids": unresolvable,
        "unresolvable_count": len(unresolvable),
        "has_unresolvable": bool(unresolvable) or bool(payload.get("has_unresolvable_evidence")),
        "jobs": jobs,
        "scenario_ids": ", ".join(scenario_ids) if scenario_ids else EMPTY,
        "scenario_id": str(scenario.get("scenario_id") or ""),
        "turns": _turns(scenario) if scenario else [],
        "profile_rows": _profile_rows(scenario.get("candidate_profile")) if scenario else [],
        "labels": _scale_options(payload.get("label_scale"), KIND_CLAIM),
        "flag_value": FLAG_UNRESOLVABLE_EVIDENCE,
        "flag_checked": FLAG_UNRESOLVABLE_EVIDENCE in (item.flags or ""),
    }
    view["has_evidence"] = bool(view["evidence"])
    view["has_jobs"] = bool(jobs)
    view["has_scenario"] = bool(view["turns"]) or bool(view["profile_rows"])
    view["has_turns"] = bool(view["turns"])
    return view


def item_view(item: RaterItem) -> dict[str, Any]:
    """Dispatch to the view model for this item's kind."""
    if item.kind == KIND_RELEVANCE:
        return relevance_view(item)
    if item.kind == KIND_CLAIM:
        return claim_view(item)
    raise ValueError(f"unknown item kind {item.kind!r}")
