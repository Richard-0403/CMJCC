"""Aggregate failure-path metrics over a failure-containing scenario set (R10.8/10.9).

Task 11.4. This is the integration-level counterpart to the per-case unit tests: it
builds a small experiment directory whose run bundles genuinely CONTAIN failures
(dangling evidence, a missing source, a rejected handoff, a timeout that recovered,
and one fault that slipped through undetected), loads them back through
:func:`jobrec_eval.loaders.load_bundles`, and checks the four aggregate rates added by
task 11.2.

The point of the assertions is that the rates are *informative*: over a failure-containing
set the grounding and handoff rates are strictly below ``1.000`` (R10.9) and the detection
rate is below ``1.000`` too, while a happy-path-only set reports ``None`` (N/A) for the
detection/recovery rates instead of a misleading perfect score.

Bundle artifacts are written with the same shapes ``jobrec.evaluation.exporters`` writes
(``run_record.json``, ``response_claims.json``, ``handoffs.jsonl``) using the real domain
models, and claim ``support_status`` values come from the real claim validator
(:func:`jobrec.agents.explanation_agent.validate_claims`) rather than being hand-set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.agents.explanation_agent import validate_claims
from jobrec.domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
from jobrec.domain.handoff import AgentHandoff
from jobrec.domain.recommendation import ResponseClaim
from jobrec.domain.run_record import RunRecord
from jobrec.evidence_store import EvidenceStore
from jobrec.utils.time import utcnow
from jobrec_eval.loaders import load_bundles, normalize
from jobrec_eval.metrics_extra import (
    failure_detection_rate,
    grounding_rate,
    handoff_success_rate,
    recovery_success_rate,
)
from tests.support.fault_injection import (
    make_claim,
    make_dangling_claim,
    make_unsupported_claim,
    valid_handoff_payload,
)

# Bundles must live under a recognized variant directory for ``load_bundles`` to walk them.
VARIANT = "full"


# --------------------------------------------------------------------- scenario-set model
@dataclass
class FailureCase:
    """One scenario x variant run in the failure-containing set.

    ``claims`` are the claims RECORDED for the run (post-validation, so flagged
    ``unsupported`` claims are included) and ``handoffs`` the handoff attempts. The four
    boolean flags are the ``run_metrics`` columns the detection/recovery rates read.
    """

    scenario_id: str
    claims: list[ResponseClaim] = field(default_factory=list)
    handoffs: list[AgentHandoff] = field(default_factory=list)
    failure_injected: bool = False
    failure_detected: bool = False
    recoverable: bool = False
    recovered: bool = False
    success: bool = True

    @property
    def run_id(self) -> str:
        return f"run-{self.scenario_id}"


def _registered_claim(store: EvidenceStore, field_name: str, value: object, text: str) -> ResponseClaim:
    """A claim bound to an evidence id that really is registered in ``store``."""
    item = store.register_field(
        EvidenceSource.JOB_POSTING,
        "job-1",
        field_name,
        value,
        confidence=1.0,
        confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.ACTIVE_SEARCH,
    )
    return make_claim(claim_type="job_attribute", text=text, evidence_ids=[item.evidence_id])


def _handoff(scenario_id: str, index: int, **overrides) -> AgentHandoff:
    """A real ``AgentHandoff`` record for this run (valid, failed or recovered)."""
    payload = valid_handoff_payload(
        handoff_id=f"handoff-{scenario_id}-{index}",
        run_id=f"run-{scenario_id}",
        **overrides,
    )
    return AgentHandoff(**payload)


def _completed_handoff(scenario_id: str, index: int = 1) -> AgentHandoff:
    return _handoff(scenario_id, index, validation_passed=True, status="completed",
                    completed_at=utcnow())


def failure_scenario_set() -> list[FailureCase]:
    """A six-scenario set containing genuine grounding, handoff and recovery failures.

    Hand-calculable totals across the set:

    * claims: 10 factual, 8 supported          -> grounding_rate      = 0.80
    * handoffs: 7 attempted, 4 valid+completed -> handoff_success_rate = 4/7
    * injected: 5, detected: 4                 -> failure_detection_rate = 0.80
    * recoverable: 2, recovered: 1             -> recovery_success_rate  = 0.50
    """
    store = EvidenceStore()

    def validated(*claims: ResponseClaim) -> list[ResponseClaim]:
        """Run the real claim validator; record supported AND flagged claims."""
        supported, dropped = validate_claims(list(claims), store)
        return supported + dropped

    cases = [
        # 1) Happy path: every claim grounded, handoff completed, no fault injected.
        FailureCase(
            scenario_id="sc-happy-path",
            claims=validated(
                _registered_claim(store, "salary_min", 4500.0, "This role pays from RM4500."),
                _registered_claim(store, "location", "Kuala Lumpur", "It is based in Kuala Lumpur."),
            ),
            handoffs=[_completed_handoff("sc-happy-path")],
        ),
        # 2) Dangling evidence id: the validator flags the claim unsupported (R10.1/10.6).
        FailureCase(
            scenario_id="sc-dangling-evidence",
            claims=validated(
                _registered_claim(store, "work_modes", "hybrid", "The role is hybrid."),
                make_dangling_claim(text="The team is famously supportive."),
            ),
            handoffs=[_completed_handoff("sc-dangling-evidence")],
            failure_injected=True,
            failure_detected=True,
        ),
        # 3) Missing source: a claim with no evidence ids at all (R10.2/10.6).
        FailureCase(
            scenario_id="sc-missing-source",
            claims=validated(
                _registered_claim(store, "employment_type", "full_time", "It is a full-time post."),
                make_unsupported_claim(text="The posting includes a signing bonus."),
            ),
            handoffs=[_completed_handoff("sc-missing-source")],
            failure_injected=True,
            failure_detected=True,
        ),
        # 4) Rejected handoff: validation failed, so the run is not a success (R10.3/10.7).
        FailureCase(
            scenario_id="sc-invalid-handoff",
            claims=validated(
                _registered_claim(store, "seniority", "junior", "It targets junior candidates."),
            ),
            handoffs=[_handoff("sc-invalid-handoff", 1, validation_passed=False,
                               status="failed", error_code="handoff_schema_invalid")],
            failure_injected=True,
            failure_detected=True,
            success=False,
        ),
        # 5) Timeout with retry: the first handoff attempt recovered, the retry completed (R10.4).
        FailureCase(
            scenario_id="sc-timeout-retry",
            claims=validated(
                _registered_claim(store, "skills", "python", "Python is listed as required."),
                _registered_claim(store, "industry", "technology", "The employer is a tech firm."),
            ),
            handoffs=[
                _handoff("sc-timeout-retry", 1, validation_passed=True, status="recovered",
                         error_code="llm_timeout"),
                _completed_handoff("sc-timeout-retry", 2),
            ],
            failure_injected=True,
            failure_detected=True,
            recoverable=True,
            recovered=True,
        ),
        # 6) Recoverable fault that was neither detected nor recovered: the handoff never
        #    completed, so detection and recovery are both below 1.000.
        FailureCase(
            scenario_id="sc-undetected-partial-failure",
            claims=validated(
                _registered_claim(store, "remote_policy", "onsite", "The role is onsite."),
            ),
            handoffs=[_handoff("sc-undetected-partial-failure", 1, validation_passed=False,
                               status="attempted")],
            failure_injected=True,
            failure_detected=False,
            recoverable=True,
            recovered=False,
            success=False,
        ),
    ]
    return cases


# ------------------------------------------------------------------------ bundle writing
def _run_record(case: FailureCase) -> RunRecord:
    return RunRecord(
        run_id=case.run_id,
        scenario_id=case.scenario_id,
        session_id=f"sess-{case.scenario_id}",
        candidate_id="cand-failure-set",
        experiment_variant=VARIANT,
        handoff_ids=[h.handoff_id for h in case.handoffs],
        started_at=utcnow(),
        completed_at=utcnow(),
        success=case.success,
        failure_code=None if case.success else "handoff_invalid",
        config_hash="cfg-hash",
        catalog_hash="cat-hash",
        prompt_hash="prompt-hash",
        code_version="test",
    )


def write_experiment(root: Path, cases: list[FailureCase]) -> Path:
    """Write ``{variant}/{scenario_id}/0/`` bundles for ``cases`` and return the experiment dir."""
    for case in cases:
        run_dir = root / VARIANT / case.scenario_id / "0"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_record.json").write_text(
            json.dumps(_run_record(case).model_dump(mode="json"), indent=2, default=str)
        )
        (run_dir / "response_claims.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in case.claims], indent=2, default=str)
        )
        with (run_dir / "handoffs.jsonl").open("w") as fh:
            for handoff in case.handoffs:
                fh.write(json.dumps(handoff.model_dump(mode="json"), default=str) + "\n")
    return root


def run_metrics_frame(cases: list[FailureCase]) -> pd.DataFrame:
    """The ``run_metrics`` rows the detection / recovery rates consume."""
    return pd.DataFrame([{
        "run_id": case.run_id,
        "scenario_id": case.scenario_id,
        "variant": VARIANT,
        "success_run": case.success,
        "failure_injected": case.failure_injected,
        "failure_detected": case.failure_detected,
        "recoverable": case.recoverable,
        "recovered": case.recovered,
    } for case in cases])


@pytest.fixture(scope="module")
def failure_set(tmp_path_factory) -> tuple[list, pd.DataFrame]:
    """Bundles loaded from disk plus the matching run-metrics frame."""
    cases = failure_scenario_set()
    exp_dir = write_experiment(tmp_path_factory.mktemp("exp-failure-metrics"), cases)
    return load_bundles(exp_dir), run_metrics_frame(cases)


# ------------------------------------------------------------------------------- tests
def test_grounding_and_handoff_rates_are_below_one_over_failure_set(failure_set):
    """Over a failure-containing set both rates are informative, not fixed at 1.000 (R10.9)."""
    bundles, _ = failure_set
    assert len(bundles) == 6, "every scenario in the failure set must load as a bundle"

    # 10 factual claims recorded, 8 of which the validator marked supported.
    assert grounding_rate(bundles) == pytest.approx(0.8)
    assert grounding_rate(bundles) < 1.0
    # 7 handoff attempts, 4 both validated and completed (failed / attempted / recovered
    # attempts never count as successes).
    assert handoff_success_rate(bundles) == pytest.approx(4 / 7)
    assert handoff_success_rate(bundles) < 1.0


def test_detection_and_recovery_rates_over_failure_set(failure_set):
    """detected/injected and recovered/recoverable are reported as measured (R10.8)."""
    _, run_metrics = failure_set

    # 5 injected faults, 4 detected -- one slipped through, so detection is not 1.000 either.
    assert failure_detection_rate(run_metrics) == pytest.approx(0.8)
    assert failure_detection_rate(run_metrics) < 1.0
    # 2 recoverable faults, 1 actually recovered.
    assert recovery_success_rate(run_metrics) == pytest.approx(0.5)


def test_rates_are_stable_across_a_csv_round_trip(failure_set, tmp_path):
    """Reloading run_metrics from CSV (booleans as text) yields the same rates."""
    _, run_metrics = failure_set
    csv_path = tmp_path / "run_metrics.csv"
    run_metrics.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path, dtype=str)

    assert failure_detection_rate(reloaded) == pytest.approx(0.8)
    assert recovery_success_rate(reloaded) == pytest.approx(0.5)


def test_happy_path_only_set_reports_na_instead_of_a_perfect_score(failure_set):
    """With no injected faults the detection/recovery rates are N/A, not a misleading 1.000."""
    bundles, run_metrics = failure_set
    happy_bundles = [b for b in bundles if b.scenario_id == "sc-happy-path"]
    happy_metrics = run_metrics[run_metrics["scenario_id"] == "sc-happy-path"]

    assert failure_detection_rate(happy_metrics) is None
    assert recovery_success_rate(happy_metrics) is None
    # The grounding/handoff rates DO read 1.000 here -- which is exactly why the
    # failure-containing set above is the one used for the R10.9 claim.
    assert grounding_rate(happy_bundles) == pytest.approx(1.0)
    assert handoff_success_rate(happy_bundles) == pytest.approx(1.0)


def test_missing_columns_and_empty_bundles_report_na(failure_set):
    """Absent instrumentation columns / no claims or handoffs report N/A, never 0.0 or 1.0."""
    _, run_metrics = failure_set

    assert failure_detection_rate(run_metrics.drop(columns=["failure_detected"])) is None
    assert failure_detection_rate(run_metrics.drop(columns=["failure_injected"])) is None
    assert recovery_success_rate(run_metrics.drop(columns=["recovered"])) is None
    assert recovery_success_rate(run_metrics.drop(columns=["recoverable"])) is None
    assert grounding_rate([]) is None
    assert handoff_success_rate([]) is None


def test_aggregate_rates_match_the_normalized_tables(failure_set):
    """The aggregate rates agree with the normalized claim / handoff CSV tables."""
    bundles, _ = failure_set
    tables = normalize(bundles)

    claims = tables["claims"]
    assert len(claims) == 10
    assert grounding_rate(bundles) == pytest.approx(claims["supported_binary"].mean())

    handoffs = tables["handoffs"]
    valid = handoffs[handoffs["validation_passed"] & (handoffs["status"] == "completed")]
    assert handoff_success_rate(bundles) == pytest.approx(len(valid) / len(handoffs))

# ------------------------------------------------------- Property 20 generated failure sets
# Kinds of run a generated scenario set may contain. Every kind contributes exactly one
# grounded claim and one handoff attempt; the *grounding* failure kinds add one extra claim
# that cannot be grounded, and the *handoff* failure kinds leave their handoff short of a
# validated ``completed`` status. That makes the expected rates hand-computable from the
# generated kinds alone, independently of the metric implementations.
_GROUNDING_FAILURE_KINDS = ("dangling_evidence", "missing_source")
_HANDOFF_FAILURE_KINDS = ("invalid_handoff", "attempted_handoff", "recovered_handoff")
_CASE_KINDS = ("happy", *_GROUNDING_FAILURE_KINDS, *_HANDOFF_FAILURE_KINDS)

# Cycled so each generated run grounds its claim on a different registered job field.
_GROUNDED_FIELDS = (
    ("salary_min", 4200.0),
    ("location", "Kuala Lumpur"),
    ("work_modes", "hybrid"),
    ("employment_type", "full_time"),
    ("seniority", "junior"),
    ("skills", "python"),
)


def _property_case(store: EvidenceStore, index: int, kind: str) -> FailureCase:
    """One generated run: always one grounded claim and one handoff, plus ``kind``'s fault."""
    field_name, value = _GROUNDED_FIELDS[index % len(_GROUNDED_FIELDS)]
    scenario_id = f"sc-{index:02d}-{kind.replace('_', '-')}"

    raw_claims = [
        _registered_claim(store, field_name, value, f"Run {index}: {field_name} is {value}."),
    ]
    if kind == "dangling_evidence":
        raw_claims.append(make_dangling_claim(text=f"Run {index}: evidence never registered."))
    elif kind == "missing_source":
        raw_claims.append(make_unsupported_claim(text=f"Run {index}: no source at all."))
    # support_status comes from the real validator, never hand-set.
    supported, dropped = validate_claims(raw_claims, store)

    if kind == "invalid_handoff":
        handoff = _handoff(scenario_id, 1, validation_passed=False, status="failed",
                           error_code="handoff_schema_invalid")
    elif kind == "attempted_handoff":
        handoff = _handoff(scenario_id, 1, validation_passed=False, status="attempted")
    elif kind == "recovered_handoff":
        handoff = _handoff(scenario_id, 1, validation_passed=True, status="recovered",
                           error_code="llm_timeout")
    else:
        handoff = _completed_handoff(scenario_id)

    return FailureCase(
        scenario_id=scenario_id,
        claims=supported + dropped,
        handoffs=[handoff],
        failure_injected=kind != "happy",
        failure_detected=kind != "happy",
        success=kind not in _HANDOFF_FAILURE_KINDS,
    )


# Feature: cmjcc-experiment-readiness, Property 20: Grounding and handoff rates are below 1.0
# over failure-containing sets
@settings(max_examples=100, deadline=None)
@given(kinds=st.lists(st.sampled_from(_CASE_KINDS), min_size=1, max_size=5))
def test_property_grounding_and_handoff_rates_below_one_over_failure_sets(
    tmp_path_factory, kinds
) -> None:
    """Any set containing a genuine grounding / handoff failure scores strictly below 1.000.

    The generated sets mix happy-path runs with grounding failures (dangling evidence id,
    missing source) and handoff failures (rejected, still-attempted, recovered-but-never-
    completed) in any proportion. Bundles are written and re-loaded through the real
    ``load_bundles`` path, and claim ``support_status`` values come from the real
    ``validate_claims``, so the rates are measured over genuine artifacts.

    Both directions are asserted: a set with at least one grounding failure has
    ``grounding_rate < 1.0`` (and likewise ``handoff_success_rate < 1.0`` for handoff
    failures), while a set with no such failure reads exactly 1.000 -- the converse sanity
    check that keeps the strict inequality meaningful rather than vacuous.

    **Validates: Requirements 10.9**
    """
    store = EvidenceStore()
    cases = [_property_case(store, i, kind) for i, kind in enumerate(kinds)]
    exp_dir = write_experiment(tmp_path_factory.mktemp("exp-prop-20"), cases)
    bundles = load_bundles(exp_dir)
    assert len(bundles) == len(cases)

    n_grounding_failures = sum(1 for kind in kinds if kind in _GROUNDING_FAILURE_KINDS)
    n_handoff_failures = sum(1 for kind in kinds if kind in _HANDOFF_FAILURE_KINDS)
    grounding = grounding_rate(bundles)
    handoff = handoff_success_rate(bundles)

    # Hand-computable from the generated mix: one grounded claim per run, one extra
    # ungroundable claim per grounding failure, one handoff per run.
    assert grounding == pytest.approx(len(cases) / (len(cases) + n_grounding_failures))
    assert handoff == pytest.approx((len(cases) - n_handoff_failures) / len(cases))

    if n_grounding_failures:
        assert grounding < 1.0
    else:
        assert grounding == pytest.approx(1.0)

    if n_handoff_failures:
        assert handoff < 1.0
    else:
        assert handoff == pytest.approx(1.0)
