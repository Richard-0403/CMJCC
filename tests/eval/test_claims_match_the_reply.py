"""Grounding is reported over the claims, so the claims and the reply must be the same text.

The defect
----------
460 of 3578 delivered claims in a 210-run deterministic batch never appeared in
``response.message``: 388 ``candidate_preference``, 36 ``no_match_reason`` and 36
``no_match_cause``. The recommendation rendered a compact summary ("Based on your request:
data analyst, in Kuala Lumpur, salary >= 4000") while the claims were separate per-field
sentences, and the no-match path appended its reasons to the claim list and to nothing else.

SC-E-02 is the clearest case. The reply showed only:

    Blocking condition: preferred_locations

while the claim being scored for grounding said:

    42 of the 50 job(s) in scope did not meet your requirement on preferred locations.

So the reported grounding rate described text the user never read, and the obvious question --
"is the paper evaluating the explanation, or an internal artifact of it?" -- had no good
answer.

Both directions matter
----------------------
* Every DELIVERED claim's sentence must be visible, or the metric grades hidden text.
* Every factual sentence must be backed by a delivered claim, or the reply asserts something
  the validator rejected. This was the worse half: dropping a claim left its sentence on
  screen and merely stopped counting it, so a rejected assertion was still shown.

The fix renders the message AFTER validation and withdraws the sentence of any dropped claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios.jsonl"

#: Lines the reply may contain WITHOUT a claim behind them, because they assert no fact about
#: the candidate or a job. Everything else must be claim-backed.
_STRUCTURAL_PREFIXES = (
    "Based on your request:",
    "No job in the current search scope",
    "  - Jobs evaluated in that scope:",
    "  - Blocking condition:",
    "You can refine any preference",
    "You could relax",
    "No eligible jobs were found.",
    "  - Note: this posting does not state a salary.",
    "#",  # the job header line: "#1 Title @ Company (match 0.87)"
)


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, base_dir="configs")


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    return [json.loads(line) for line
            in Path(SCENARIOS).read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(config, scenario: dict):
    service = AppService(config, CATALOG)
    candidate = service.create_candidate(scenario["profile"])
    session = service.create_session(candidate.candidate_id, "full")
    result = None
    for text in scenario["turns"]:
        result = service.process_turn(session, text, scenario_id=scenario["scenario_id"])
    return result


def _is_structural(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(line.startswith(p) or stripped.startswith(p.strip())
               for p in _STRUCTURAL_PREFIXES)


# ------------------------------------------- forward: every delivered claim is visible
def test_every_delivered_claim_appears_in_the_reply(config, scenarios):
    """Across all 42 authoritative scenarios, not a sample."""
    missing: list[str] = []
    checked = 0
    for scenario in scenarios:
        result = _run(config, scenario)
        message = result.response.message
        for claim in result.response.claims:
            checked += 1
            if claim.text not in message:
                missing.append(f"{scenario['scenario_id']} [{claim.claim_type}] {claim.text}")

    assert checked, "no claims were delivered at all, so this proved nothing"
    assert not missing, (
        f"{len(missing)} of {checked} delivered claims are absent from the reply:\n  "
        + "\n  ".join(missing[:10]))


def test_the_no_match_reply_shows_its_reasons(config, scenarios):
    """The SC-E-02 case: reasons were computed, scored, and never displayed."""
    seen = 0
    for scenario in scenarios:
        result = _run(config, scenario)
        if not (result.decision and result.decision.no_match):
            continue
        seen += 1
        for claim in result.response.claims:
            if claim.claim_type in ("no_match_reason", "no_match_cause"):
                assert claim.text in result.response.message, (
                    f"{scenario['scenario_id']}: {claim.claim_type} was scored but not shown")
    assert seen, "no no-match scenario ran, so this proved nothing"


# --------------------------------- reverse: every factual sentence is claim-backed
def test_no_factual_sentence_survives_without_a_delivered_claim(config, scenarios):
    """A rejected claim's sentence must be withdrawn, not merely uncounted."""
    orphans: list[str] = []
    for scenario in scenarios:
        result = _run(config, scenario)
        # A clarification response ASKS rather than asserts, so it carries no factual claim by
        # design. Scoping this by response type rather than by matching the question's wording,
        # which would only pin the current phrasing.
        if result.response.response_type == "clarification":
            assert not result.response.claims, (
                f"{scenario['scenario_id']}: a clarification response made factual claims")
            continue
        texts = {c.text for c in result.response.claims}
        for line in result.response.message.split("\n"):
            if _is_structural(line):
                continue
            if not any(t in line for t in texts):
                orphans.append(f"{scenario['scenario_id']}: {line!r}")

    assert not orphans, (
        f"{len(orphans)} factual line(s) have no delivered claim behind them:\n  "
        + "\n  ".join(orphans[:10]))


def test_a_dropped_claims_sentence_is_withdrawn_from_the_reply(config):
    """Directly: force a claim to fail validation and check the line disappears.

    A ``skill_gap`` citing only the job's requirement cannot establish an absence in the
    candidate's record, so it is dropped -- and its sentence must go with it.
    """
    from jobrec.agents.explanation_agent import ExplanationAgent, validate_claims
    from jobrec.evidence_store import EvidenceStore

    store = EvidenceStore()
    agent = ExplanationAgent(store, config)
    claim = agent._claim(
        "skill_gap", "Gap: the role requires excel, which is not recorded in your profile "
        "skills.", ["ev-missing"], "job-1",
        predicate="skill_not_recorded", job_id="job-1", claim_args={"skill": "excel"})
    lines = ["Based on your request:", f"  - {claim.text}", "You can refine any preference."]
    line_of = {claim.claim_id: f"  - {claim.text}"}

    delivered, dropped = validate_claims([claim], store)
    assert delivered == [] and len(dropped) == 1

    message = agent._render(lines, line_of, dropped)
    assert claim.text not in message
    assert "Based on your request:" in message
    assert "You can refine any preference." in message
