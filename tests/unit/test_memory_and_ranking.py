"""Unit tests for memory conflicts, ranking invariants and claim validation."""

from __future__ import annotations

from jobrec.agents.candidate_understanding import CandidateUnderstandingAgent
from jobrec.agents.explanation_agent import validate_claims
from jobrec.agents.memory_agent import MemoryAgent
from jobrec.config import load_config
from jobrec.domain.dialogue import DialogueState
from jobrec.domain.recommendation import ResponseClaim
from jobrec.evidence_store import EvidenceStore
from jobrec.orchestration.cmjcc import CMJCC, CMJCCInput


def _cmjcc_run(profile, text, cfg):
    store = EvidenceStore()
    mem = MemoryAgent(store, cfg)
    cand = mem.create_candidate_state(profile)
    dlg = DialogueState(session_id="s", candidate_id=profile["candidate_id"], version=1, turns=[])
    ex = CandidateUnderstandingAgent().extract(text)
    dlg = mem.append_turn(dlg, "candidate", text)
    out = CMJCC(store, cfg).run(CMJCCInput(cand, dlg, ex, "snap", cfg, "run"))
    return cand, out, store


def test_temporary_override_does_not_pollute_long_term(config):
    cand, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "preferred_locations": ["Penang"]},
        "I want a data analyst role in Kuala Lumpur now.", config,
    )
    # long-term profile keeps Penang
    assert [p.value for p in cand.preferred_locations] == ["Penang"]
    # active search uses Kuala Lumpur
    assert out.active_search_state.preferred_locations == ["Kuala Lumpur"]


def test_factual_years_conflict_triggers_clarification(config):
    _, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "years_experience": 1, "target_roles": ["Data Analyst"]},
        "Actually I have 3 years experience.", config,
    )
    assert any(c.field_name == "years_experience" and c.resolution == "ask_clarification"
               for c in out.conflicts)
    assert out.active_search_state.years_experience == 1.0  # not silently overwritten


def test_work_mode_merges(config):
    _, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "work_modes": ["remote"], "target_roles": ["Data Analyst"]},
        "hybrid is also fine", config,
    )
    assert set(out.active_search_state.work_modes) == {"remote", "hybrid"}


def test_claim_validator_drops_unsupported(store):
    from jobrec.domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope

    item = store.register_field(
        source=EvidenceSource.PROFILE, source_object_id="c", field_name="skills",
        normalized_value="python", confidence=1.0,
        confirmation=ConfirmationStatus.CONFIRMED, scope=PersistenceScope.LONG_TERM,
    )
    supported = ResponseClaim(claim_id="ok", claim_type="candidate_preference",
                              text="knows python", evidence_ids=[item.evidence_id])
    bad = ResponseClaim(claim_id="bad", claim_type="job_attribute",
                        text="great culture", evidence_ids=["does-not-exist"])
    keep, drop = validate_claims([supported, bad], store)
    assert [c.claim_id for c in keep] == ["ok"]
    assert [c.claim_id for c in drop] == ["bad"]


def test_config_hash_stable_and_variant_sensitive():
    a = load_config("configs/experiment_full.yaml", base_dir="configs")
    b = load_config("configs/experiment_full.yaml", base_dir="configs")
    c = load_config("configs/experiment_no_context.yaml", base_dir="configs")
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
