"""Data-quality validation for the job catalog and scenario set (R17).

Malformed inputs must be caught *before* an experiment runs, so this module
validates the two inputs the whole pipeline depends on -- the normalised job
catalog and the tagged scenario set -- and emits a machine-readable
``data_quality_report.json`` in which every finding names the offending
identifier and the violation type (R17.3/R17.4).

Two layers of checking, matching the requirement:

* **Catalog (R17.1)** -- duplicate job ids, salary ranges where the minimum
  exceeds the maximum, unknown currencies, invalid ``work_mode`` and
  ``experience_level`` values, expired/unparseable deadlines and empty required
  fields (title, skills, location).
* **Scenarios (R17.1/R17.2)** -- duplicate scenario ids, empty turn scripts, a
  relevance label where one is required, a hard-constraint reference where one is
  required, expectation labels that contradict each other, and -- for every
  scenario labelled *no match* -- confirmation via
  :class:`~jobrec.agents.job_context_agent.JobContextAgent` that no job in the
  catalog is actually eligible, so a mislabelled scenario cannot masquerade as a
  genuine no-match case.

Authorities, never re-implemented here:

* enums/vocabularies come from :mod:`jobrec.taxonomy`
  (``WORK_MODES``, :func:`~jobrec.taxonomy.canonical_level`,
  :func:`~jobrec.taxonomy.canonical_role`),
* the known-currency set is whatever :func:`jobrec.utils.money.can_normalize`
  accepts (the same table the catalog normaliser converts with),
* deadline parsing uses :func:`jobrec.llm.field_validation.normalize_deadline`,
* eligibility is decided by the real ``JobContextAgent`` on the full-variant
  code path, driven by the deterministic rule extractor (no LLM, no database).

Relationship to ``scripts/validate_catalog.py``: that script answers a different,
narrower question -- "does every catalog record parse against the schema?" -- by
calling :func:`jobrec.catalog.load_catalog`. It is reused rather than duplicated:
callers hand this validator already-parsed :class:`~jobrec.domain.job.JobPosting`
objects (schema validity already established by the loader), and the script grew
optional flags that delegate here for the semantic checks it never covered.
Raw ``dict`` records are also accepted so pre-normalisation data can be screened
with the same rules.

Expired-deadline nuance: a posting whose deadline has passed *and* is flagged
``is_active=False`` is expected content in the research catalog (the pipeline must
prove it never recommends it), so it is recorded as a ``warning``. A posting that
is expired while still marked active is contradictory data and is an ``error``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.agents.memory_agent import MemoryAgent
from jobrec.catalog import catalog_hash
from jobrec.config import AppConfig
from jobrec.domain.dialogue import DialogueState
from jobrec.domain.enums import ExperimentVariant
from jobrec.domain.job import JobPosting
from jobrec.evidence_store import EvidenceStore
from jobrec.llm.field_validation import normalize_deadline
from jobrec.taxonomy import WORK_MODES, canonical_level, canonical_role
from jobrec.utils.money import can_normalize

__all__ = [
    "DATA_QUALITY_REPORT_FILENAME",
    "DATA_QUALITY_VERSION",
    "DataQualityReport",
    "Finding",
    "read_data_quality_report",
    "validate_dataset",
    "write_data_quality_report",
]

#: Report file name, written at the root of the directory given to
#: :func:`write_data_quality_report`.
DATA_QUALITY_REPORT_FILENAME = "data_quality_report.json"

#: Report format version, recorded in the payload.
DATA_QUALITY_VERSION = "1.0.0"

#: Severity levels. ``error`` means the data is wrong; ``warning`` means the data
#: is usable but incomplete or deliberately degenerate (e.g. an expired posting
#: correctly flagged inactive).
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Work-mode values a catalog record may carry: the canonical taxonomy modes plus
#: the explicit "not stated" marker used by :class:`~jobrec.domain.job.JobPosting`.
_VALID_WORK_MODES: frozenset[str] = frozenset({*WORK_MODES, "unspecified"})

#: Scenario ``expects`` keys that count as naming the hard constraints a scenario
#: exercises (R17.2). Any one of them satisfies the reference requirement.
_HARD_CONSTRAINT_KEYS: tuple[str, ...] = (
    "hard_fields",
    "hard_constraint_fields",
    "blocking",
    "blocking_fields",
    "blocking_constraint",
)

#: Scenario types that assert hard-constraint behaviour and therefore need a
#: hard-constraint reference even when they expect a recommendation.
_HARD_CONSTRAINT_TYPES: frozenset[str] = frozenset({"multiple_hard"})

#: Named checks, so the report can state what ran and what could not.
CHECK_CATALOG = "catalog_records"
CHECK_DUPLICATE_JOB_IDS = "duplicate_job_ids"
CHECK_SCENARIOS = "scenario_records"
CHECK_DUPLICATE_SCENARIO_IDS = "duplicate_scenario_ids"
CHECK_RELEVANCE_LABELS = "scenario_relevance_labels"
CHECK_HARD_CONSTRAINT_REFS = "scenario_hard_constraint_references"
CHECK_NO_MATCH = "no_match_scenarios_unsatisfiable"


# --------------------------------------------------------------------- findings
@dataclass(frozen=True)
class Finding:
    """One data-quality violation, keyed by the offending identifier (R17.4)."""

    identifier: str
    entity: str
    violation_type: str
    detail: str
    severity: str = SEVERITY_ERROR
    field_name: str | None = None
    observed: Any = None

    def describe(self) -> str:
        """Human-readable one-liner naming the offender and the violation."""
        where = f" [{self.field_name}]" if self.field_name else ""
        return f"{self.severity}: {self.entity} {self.identifier}{where}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "entity": self.entity,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "field": self.field_name,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DataQualityReport:
    """Result of validating a catalog + scenario set.

    Iterating the report yields its :class:`Finding` objects, so a caller that
    only wants the design's ``list[Finding]`` can write
    ``list(validate_dataset(...))``.
    """

    findings: tuple[Finding, ...]
    job_count: int
    scenario_count: int
    reference_date: str
    checks_run: tuple[str, ...] = ()
    checks_skipped: Mapping[str, str] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == SEVERITY_WARNING)

    @property
    def ok(self) -> bool:
        """True when no ``error``-severity violation was found."""
        return not self.errors

    def by_type(self, violation_type: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.violation_type == violation_type)

    def counts_by_violation_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.violation_type] = counts.get(finding.violation_type, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        """Multi-line summary; the first line is the verdict."""
        head = (
            f"data quality {'OK' if self.ok else 'FAILED'}: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s) over "
            f"{self.job_count} job(s) and {self.scenario_count} scenario(s)"
        )
        lines = [head]
        for violation_type, count in self.counts_by_violation_type().items():
            lines.append(f"  {violation_type}: {count}")
        for check, reason in sorted(self.checks_skipped.items()):
            lines.append(f"  not checked: {check} ({reason})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable payload written to ``data_quality_report.json`` (R17.3)."""
        return {
            "report_version": DATA_QUALITY_VERSION,
            "reference_date": self.reference_date,
            "job_count": self.job_count,
            "scenario_count": self.scenario_count,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks_run": list(self.checks_run),
            "checks_skipped": dict(self.checks_skipped),
            "counts_by_violation_type": self.counts_by_violation_type(),
            "findings": [f.to_dict() for f in self.findings],
        }


# ------------------------------------------------------------------- input views
@dataclass(frozen=True)
class _JobView:
    """Uniform read model over a :class:`JobPosting` or a raw catalog dict."""

    index: int
    job_id: str
    title: str
    skills: list[str]
    locations: list[str]
    salary_min: Any
    salary_max: Any
    currency: Any
    work_mode: Any
    experience_level: Any
    deadline: Any
    is_active: bool

    @property
    def label(self) -> str:
        return self.job_id or f"<catalog row {self.index}>"


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _job_view(job: JobPosting | Mapping[str, Any], index: int) -> _JobView:
    if isinstance(job, JobPosting):
        return _JobView(
            index=index,
            job_id=job.job_id,
            title=job.title,
            skills=[*job.required_skills, *job.preferred_skills],
            locations=[v for v in (job.city, job.region, job.country) if v],
            salary_min=job.salary_min,
            salary_max=job.salary_max,
            currency=job.salary_currency,
            work_mode=job.work_mode,
            experience_level=job.experience_level,
            deadline=job.application_deadline,
            is_active=job.is_active,
        )
    data = dict(job)
    return _JobView(
        index=index,
        job_id=str(data.get("job_id") or "").strip(),
        title=str(data.get("title") or "").strip(),
        skills=[*_as_list(data.get("required_skills")), *_as_list(data.get("preferred_skills"))],
        locations=[
            str(data[key]).strip()
            for key in ("city", "region", "country")
            if str(data.get(key) or "").strip()
        ],
        salary_min=data.get("salary_min"),
        salary_max=data.get("salary_max"),
        currency=data.get("salary_currency"),
        work_mode=data.get("work_mode"),
        experience_level=data.get("experience_level"),
        deadline=data.get("application_deadline"),
        is_active=bool(data.get("is_active", True)),
    )


@dataclass(frozen=True)
class _ScenarioView:
    """Uniform read model over a :class:`~jobrec_eval.scenarios.Scenario` or dict."""

    index: int
    scenario_id: str
    scenario_type: str
    expected_response: str
    no_match_expected: bool
    clarification_expected: bool
    turns: list[str]
    profile: dict[str, Any]
    expects: dict[str, Any]

    @property
    def label(self) -> str:
        return self.scenario_id or f"<scenario row {self.index}>"

    @property
    def expects_no_match(self) -> bool:
        return self.no_match_expected or self.expected_response == "no_match"

    @property
    def expects_clarification(self) -> bool:
        """True when the scenario's turn script ends in a clarification.

        Read from the ``clarification_expected`` tag *or* an ``expects`` block that
        names ``clarification`` as the response type, since not every scenario set
        carries the boolean tag.
        """
        return self.clarification_expected or self.expected_response == "clarification"

    @property
    def has_hard_constraint_reference(self) -> bool:
        return any(self.expects.get(key) for key in _HARD_CONSTRAINT_KEYS)

    @property
    def needs_hard_constraint_reference(self) -> bool:
        return self.expects_no_match or self.scenario_type in _HARD_CONSTRAINT_TYPES

    @property
    def needs_relevance_label(self) -> bool:
        """Recommendation-returning scenarios need a graded reference to be scored."""
        return not self.expects_no_match and not self.expects_clarification


def _scenario_view(scenario: Any, index: int) -> _ScenarioView:
    if isinstance(scenario, Mapping):
        expects = dict(scenario.get("expects") or {})
        return _ScenarioView(
            index=index,
            scenario_id=str(scenario.get("scenario_id") or "").strip(),
            scenario_type=str(scenario.get("scenario_type") or "unknown"),
            expected_response=str(expects.get("response_type") or "recommendation"),
            no_match_expected=bool(scenario.get("no_match_expected", False)),
            clarification_expected=bool(scenario.get("clarification_expected", False)),
            turns=[str(t) for t in (scenario.get("turns") or [])],
            profile=dict(scenario.get("profile") or {}),
            expects=expects,
        )
    expects = dict(getattr(scenario, "expects", {}) or {})
    return _ScenarioView(
        index=index,
        scenario_id=str(getattr(scenario, "scenario_id", "") or "").strip(),
        scenario_type=str(getattr(scenario, "scenario_type", "unknown")),
        expected_response=str(getattr(scenario, "expected_response", "recommendation")),
        no_match_expected=bool(getattr(scenario, "no_match_expected", False)),
        clarification_expected=bool(getattr(scenario, "clarification_expected", False)),
        turns=[str(t) for t in (getattr(scenario, "turns", []) or [])],
        profile=dict(getattr(scenario, "profile", {}) or {}),
        expects=expects,
    )


def _scenario_views(scenarios: Any) -> list[_ScenarioView]:
    """Normalise any supported scenario container into ordered views.

    Accepts a mapping keyed by scenario id (as :func:`jobrec_eval.scenarios.load_scenarios`
    returns) or any sequence of ``Scenario`` objects / raw dicts.
    """
    if isinstance(scenarios, Mapping):
        items: Iterable[Any] = list(scenarios.values())
    else:
        items = list(scenarios)
    return [_scenario_view(item, index) for index, item in enumerate(items)]


# ----------------------------------------------------------------- catalog checks
def _catalog_findings(jobs: Sequence[JobPosting | Mapping[str, Any]],
                      reference_date: date) -> list[Finding]:
    views = [_job_view(job, index) for index, job in enumerate(jobs)]
    findings: list[Finding] = []

    # Duplicate (and empty) job ids.
    seen: dict[str, int] = {}
    for view in views:
        if not view.job_id:
            findings.append(Finding(
                identifier=view.label, entity="job", violation_type="empty_job_id",
                field_name="job_id", detail="catalog record has no job_id",
            ))
            continue
        seen[view.job_id] = seen.get(view.job_id, 0) + 1
        if seen[view.job_id] > 1:
            findings.append(Finding(
                identifier=view.job_id, entity="job", violation_type="duplicate_job_id",
                field_name="job_id", observed=seen[view.job_id],
                detail=f"job_id appears {seen[view.job_id]} times in the catalog",
            ))

    for view in views:
        findings.extend(_job_field_findings(view, reference_date))
    return findings


def _job_field_findings(view: _JobView, reference_date: date) -> list[Finding]:
    findings: list[Finding] = []

    def add(violation_type: str, detail: str, *, field_name: str | None = None,
            observed: Any = None, severity: str = SEVERITY_ERROR) -> None:
        findings.append(Finding(
            identifier=view.label, entity="job", violation_type=violation_type,
            field_name=field_name, observed=observed, detail=detail, severity=severity,
        ))

    # --- salary range -------------------------------------------------------
    low, high = _as_number(view.salary_min), _as_number(view.salary_max)
    if low is not None and high is not None and low > high:
        add("salary_min_exceeds_max", f"salary_min {low} exceeds salary_max {high}",
            field_name="salary_min", observed={"salary_min": low, "salary_max": high})

    # --- currency -----------------------------------------------------------
    currency = str(view.currency).strip() if view.currency not in (None, "") else ""
    if currency and not can_normalize(currency, "month"):
        add("unknown_currency", f"currency '{currency}' has no known conversion",
            field_name="salary_currency", observed=currency)

    # --- enums --------------------------------------------------------------
    work_mode = str(view.work_mode).strip() if view.work_mode not in (None, "") else ""
    if work_mode and work_mode.lower() not in _VALID_WORK_MODES:
        add("invalid_work_mode", f"work_mode '{work_mode}' is not a valid value",
            field_name="work_mode", observed=work_mode)

    level = str(view.experience_level).strip() if view.experience_level not in (None, "") else ""
    if level and canonical_level(level) is None:
        add("invalid_experience_level",
            f"experience_level '{level}' is not in the taxonomy",
            field_name="experience_level", observed=level)

    # --- deadline -----------------------------------------------------------
    if view.deadline not in (None, ""):
        iso, warnings = normalize_deadline(view.deadline)
        if iso is None:
            add("invalid_deadline", warnings[0] if warnings else "deadline is unparseable",
                field_name="application_deadline", observed=str(view.deadline))
        elif date.fromisoformat(iso) < reference_date:
            if view.is_active:
                add("expired_deadline_active",
                    f"deadline {iso} passed before the reference date "
                    f"{reference_date.isoformat()} while the posting is still active",
                    field_name="application_deadline", observed=iso)
            else:
                add("expired_deadline",
                    f"deadline {iso} passed before the reference date "
                    f"{reference_date.isoformat()} (posting is flagged inactive)",
                    field_name="application_deadline", observed=iso,
                    severity=SEVERITY_WARNING)

    # --- empty required fields ---------------------------------------------
    if not view.title.strip():
        add("empty_title", "title is empty", field_name="title")
    if not view.skills:
        add("empty_skills", "no required or preferred skills", field_name="required_skills")
    if not view.locations:
        add("empty_location", "no city, region or country", field_name="city")
    return findings


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- scenario checks
def _scenario_findings(
    views: Sequence[_ScenarioView],
    label_ids: set[str] | None,
) -> list[Finding]:
    findings: list[Finding] = []

    seen: dict[str, int] = {}
    for view in views:
        if not view.scenario_id:
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="empty_scenario_id", field_name="scenario_id",
                detail="scenario record has no scenario_id",
            ))
        else:
            seen[view.scenario_id] = seen.get(view.scenario_id, 0) + 1
            if seen[view.scenario_id] > 1:
                findings.append(Finding(
                    identifier=view.scenario_id, entity="scenario",
                    violation_type="duplicate_scenario_id", field_name="scenario_id",
                    observed=seen[view.scenario_id],
                    detail=f"scenario_id appears {seen[view.scenario_id]} times",
                ))

        if not view.turns:
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="empty_scenario_turns", field_name="turns",
                detail="scenario has no candidate turns to run",
            ))

        if view.no_match_expected and view.expected_response not in ("", "no_match"):
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="inconsistent_no_match_expectation",
                field_name="expects.response_type",
                observed=view.expected_response,
                detail=("no_match_expected is true but the expected response type is "
                        f"'{view.expected_response}'"),
            ))

        if view.needs_hard_constraint_reference and not view.has_hard_constraint_reference:
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="missing_hard_constraint_reference",
                field_name="expects", severity=SEVERITY_WARNING,
                detail=("scenario asserts hard-constraint behaviour but names no "
                        "hard/blocking constraint reference"),
            ))

        if label_ids is not None and view.needs_relevance_label:
            if view.scenario_id not in label_ids:
                findings.append(Finding(
                    identifier=view.label, entity="scenario",
                    violation_type="missing_relevance_label",
                    field_name="relevance_grade", severity=SEVERITY_WARNING,
                    detail="scenario returns recommendations but has no relevance label",
                ))
    return findings


# -------------------------------------------------------- no-match verification
def _no_match_findings(
    views: Sequence[_ScenarioView],
    catalog: Sequence[JobPosting],
    config: AppConfig,
    catalog_snapshot_id: str,
) -> tuple[list[Finding], dict[str, str]]:
    """Confirm each no-match scenario really has no eligible job (R17.2).

    Replays the scenario's turns through the real orchestration path in its
    ``full``-variant, deterministic (rule-extractor, no LLM, no database) form to
    obtain the authoritative constraint bundle and active search, then evaluates
    **every** catalog job against it with :class:`JobContextAgent` -- the whole
    catalog, not the retrieval pool, so the verdict does not depend on recall.

    Eligible jobs are graded by role fit using the same rule as the relevance
    oracle (canonical role-family match):

    * an eligible job in a role family the scenario asked for makes the scenario
      genuinely satisfiable, so the no-match label is wrong (``error``);
    * an eligible job in another role family means the stated constraints alone
      are satisfiable and the no-match outcome rests on role fit / retrieval
      rather than joint infeasibility (``warning``).
    """
    targets = [view for view in views if view.expects_no_match]
    findings: list[Finding] = []
    skipped: dict[str, str] = {}
    if not targets:
        # Nothing to verify: the check is satisfied vacuously, not skipped.
        return findings, skipped
    if not catalog:
        skipped[CHECK_NO_MATCH] = "no catalog supplied to evaluate eligibility against"
        return findings, skipped

    cfg = config.model_copy(deep=True)
    # The no-match claim is only meaningful under explicit constraint
    # orchestration, i.e. the full variant; never under a no_context ablation.
    cfg.experiment.variant = ExperimentVariant.FULL
    cfg.context.explicit_constraint_orchestration = True
    agent = JobContextAgent(cfg)
    cat_hash = catalog_hash(list(catalog))

    unrunnable: list[str] = []
    for view in targets:
        if not view.turns or not view.profile:
            unrunnable.append(view.label)
            continue
        context, active = _replay_scenario(view, catalog, cfg, catalog_snapshot_id, cat_hash)
        if context is None or active is None:
            unrunnable.append(view.label)
            continue
        eligible = [job for job in catalog if agent.evaluate(job, context).eligible]
        if not eligible:
            continue
        wanted = {canonical_role(role) for role in active.target_roles}
        on_role = [
            job.job_id for job in eligible
            if (job.role_family or canonical_role(job.title)) in wanted
        ]
        if on_role:
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="no_match_scenario_satisfiable",
                field_name="no_match_expected", observed=on_role[:10],
                detail=(f"scenario is labelled no-match but {len(on_role)} catalog job(s) "
                        "in a requested role family are eligible under its own hard "
                        "constraints"),
            ))
        else:
            findings.append(Finding(
                identifier=view.label, entity="scenario",
                violation_type="no_match_scenario_constraint_satisfiable",
                field_name="no_match_expected", severity=SEVERITY_WARNING,
                observed=[job.job_id for job in eligible][:10],
                detail=(f"{len(eligible)} catalog job(s) satisfy the scenario's hard "
                        "constraints (all outside the requested role families), so the "
                        "no-match outcome rests on role fit rather than joint "
                        "infeasibility"),
            ))
    if unrunnable:
        skipped[CHECK_NO_MATCH] = (
            "no runnable profile/turns or constraint bundle for: " + ", ".join(unrunnable)
        )
    return findings, skipped


def _replay_scenario(
    view: _ScenarioView,
    catalog: Sequence[JobPosting],
    config: AppConfig,
    catalog_snapshot_id: str,
    cat_hash: str,
) -> tuple[Any, Any]:
    """Replay a scenario's turns, returning ``(JobContextState, ActiveSearchState)``.

    Uses the shared orchestration path with ``provider=None``, which forces the
    deterministic rule extractor, so this never calls a model and never touches a
    database. Either element is ``None`` when the scenario produced no such state.
    """
    from jobrec.orchestration.orchestrator import ConversationOrchestrator

    store = EvidenceStore()
    profile = dict(view.profile)
    profile.setdefault("candidate_id", f"{view.scenario_id}-dq-cand")
    candidate = MemoryAgent(store, config).create_candidate_state(profile)
    dialogue = DialogueState(
        session_id=f"dq-{view.scenario_id or view.index}",
        candidate_id=candidate.candidate_id, version=1, turns=[],
    )
    orchestrator = ConversationOrchestrator(
        config, list(catalog), catalog_snapshot_id, cat_hash, provider=None, store=store,
    )
    context: Any = None
    active: Any = None
    for text in view.turns:
        result = orchestrator.process_turn(candidate, dialogue, text)
        candidate, dialogue = result.candidate_state, result.dialogue_state
        context = result.job_context_state or context
        active = result.active_search_state or active
    return context, active


# ------------------------------------------------------------- relevance labels
def _relevance_label_ids(source: Any) -> set[str] | None:
    """Collect the scenario ids that carry a relevance label, or ``None``.

    Accepts a CSV path, a ``pandas`` DataFrame, a mapping keyed by scenario id or
    ``(scenario_id, job_id)`` tuples, or any iterable of ids. ``None`` (or an
    unreadable/label-less source) means "not supplied", which skips the check
    rather than reporting every scenario as unlabelled.
    """
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            return None
        import pandas as pd

        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError):
            return None
        return _relevance_label_ids(frame)
    if hasattr(source, "columns"):  # pandas DataFrame
        if "scenario_id" not in source.columns:
            return None
        return {str(value) for value in source["scenario_id"].tolist()}
    if isinstance(source, Mapping):
        ids: set[str] = set()
        for key in source:
            if isinstance(key, tuple) and key:
                ids.add(str(key[0]))
            else:
                ids.add(str(key))
        return ids
    if isinstance(source, Iterable):
        return {str(value) for value in source}
    return None


# ------------------------------------------------------------------- public API
def validate_dataset(
    catalog: Sequence[JobPosting | Mapping[str, Any]],
    scenarios: Any,
    *,
    config: AppConfig | None = None,
    reference_date: str | date | None = None,
    relevance_labels: Any = None,
    verify_no_match: bool = True,
    catalog_snapshot_id: str = "catalog-unknown",
) -> DataQualityReport:
    """Validate a job catalog and scenario set, returning every finding (R17.1-17.4).

    Args:
        catalog: Parsed :class:`~jobrec.domain.job.JobPosting` objects (as
            :func:`jobrec.catalog.load_catalog` returns) or raw catalog dicts.
        scenarios: Mapping keyed by scenario id (as
            :func:`jobrec_eval.scenarios.load_scenarios` returns) or any sequence
            of ``Scenario`` objects / raw scenario dicts.
        config: Resolved configuration; supplies the reference date and the
            constraint policies used for the no-match check. Defaults to
            :class:`~jobrec.config.AppConfig`.
        reference_date: Overrides ``config.project.reference_date`` when deciding
            which deadlines have expired.
        relevance_labels: Optional label source (CSV path, DataFrame, mapping or
            iterable of scenario ids). When omitted the relevance-label check is
            recorded as *not checked* instead of flagging every scenario.
        verify_no_match: When true (default), replay each no-match scenario and
            confirm no catalog job is eligible. Disable to skip the pipeline
            replay (e.g. for a fast, catalog-only screen).
        catalog_snapshot_id: Snapshot id stamped on the replayed constraint bundle.

    Returns:
        A :class:`DataQualityReport`. Never raises on bad data -- every problem is
        reported as a :class:`Finding`.
    """
    cfg = config or AppConfig()
    ref_date = _resolve_reference_date(reference_date, cfg)

    jobs = list(catalog)
    views = _scenario_views(scenarios)
    postings = [job for job in jobs if isinstance(job, JobPosting)]

    findings = _catalog_findings(jobs, ref_date)
    label_ids = _relevance_label_ids(relevance_labels)
    findings.extend(_scenario_findings(views, label_ids))

    checks = [
        CHECK_CATALOG, CHECK_DUPLICATE_JOB_IDS,
        CHECK_SCENARIOS, CHECK_DUPLICATE_SCENARIO_IDS,
        CHECK_HARD_CONSTRAINT_REFS,
    ]
    skipped: dict[str, str] = {}
    if label_ids is None:
        skipped[CHECK_RELEVANCE_LABELS] = "no relevance labels supplied"
    else:
        checks.append(CHECK_RELEVANCE_LABELS)

    if not verify_no_match:
        skipped[CHECK_NO_MATCH] = "no-match verification disabled by the caller"
    elif len(postings) != len(jobs):
        skipped[CHECK_NO_MATCH] = (
            "catalog contains unparsed records; supply JobPosting objects to verify"
        )
    else:
        no_match_findings, no_match_skipped = _no_match_findings(
            views, postings, cfg, catalog_snapshot_id,
        )
        findings.extend(no_match_findings)
        skipped.update(no_match_skipped)
        if CHECK_NO_MATCH not in skipped:
            checks.append(CHECK_NO_MATCH)

    return DataQualityReport(
        findings=tuple(findings),
        job_count=len(jobs),
        scenario_count=len(views),
        reference_date=ref_date.isoformat(),
        checks_run=tuple(checks),
        checks_skipped=skipped,
    )


def _resolve_reference_date(reference_date: str | date | None, config: AppConfig) -> date:
    if isinstance(reference_date, date):
        return reference_date
    if isinstance(reference_date, str) and reference_date.strip():
        return date.fromisoformat(reference_date.strip())
    return date.fromisoformat(config.project.reference_date)


def write_data_quality_report(
    report: DataQualityReport, out_dir: str | Path, filename: str = DATA_QUALITY_REPORT_FILENAME
) -> Path:
    """Write ``data_quality_report.json`` into ``out_dir`` and return its path (R17.3).

    The payload carries no timestamp, so validating unchanged inputs twice yields
    byte-identical output and the report can be checksummed with the other
    artifacts.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return target


def read_data_quality_report(out_dir: str | Path,
                             filename: str = DATA_QUALITY_REPORT_FILENAME) -> dict[str, Any]:
    """Load a previously written report payload."""
    return json.loads((Path(out_dir) / filename).read_text())
