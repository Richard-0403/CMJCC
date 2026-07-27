"""Contract tests: schema validation rejects malformed inputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobrec.domain.evidence import EvidenceItem
from jobrec.domain.extraction import ExtractedPreferenceSet
from jobrec.domain.job import JobPosting
from jobrec.domain.run_record import RunRecord
from jobrec.llm.provider import LLMInvalidJSON
from jobrec.llm.structured_output import parse_extraction


def _run_record(**overrides) -> RunRecord:
    fields = {
        "run_id": "r", "session_id": "s", "candidate_id": "c", "experiment_variant": "full",
        "started_at": "2026-01-01T00:00:00Z", "config_hash": "ch", "catalog_hash": "cath",
        "prompt_hash": "ph", "code_version": "0.1.0",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_evidence_item_requires_fields():
    with pytest.raises(ValidationError):
        EvidenceItem(evidence_id="e")  # missing required fields


def test_evidence_confidence_bounds():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="e", source="profile", source_object_id="c", field_name="skills",
            normalized_value="python", confidence=1.5, confirmation_status="confirmed",
            persistence_scope="long_term", observed_at="2026-01-01T00:00:00Z",
        )


def test_unknown_enum_rejected():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="e", source="not_a_source", source_object_id="c", field_name="skills",
            normalized_value="python", confidence=0.9, confirmation_status="confirmed",
            persistence_scope="long_term", observed_at="2026-01-01T00:00:00Z",
        )


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        JobPosting(
            job_id="j", title="t", company="c", description="d", normalized_title="t",
            source_snapshot_id="s", ingested_at="2026-01-01T00:00:00Z", raw_payload_hash="h",
            unexpected_field="boom",
        )


def test_parse_extraction_rejects_bad_json():
    with pytest.raises(LLMInvalidJSON):
        parse_extraction("{not valid json")


def test_parse_extraction_accepts_valid():
    payload = {"utterance_id": "u", "preferences": [], "detected_language": "en",
               "ambiguous_fields": [], "extraction_warnings": []}
    result = parse_extraction(payload)
    assert isinstance(result, ExtractedPreferenceSet)


def test_run_record_feature_flags_defaults_to_empty_dict():
    record = _run_record()
    assert record.feature_flags == {}
    assert record.model_dump(mode="json")["feature_flags"] == {}


def test_run_record_feature_flags_round_trip():
    flags = {"variant": "no_memory", "use_persistent_memory": False, "top_k": 5}
    dumped = _run_record(feature_flags=flags).model_dump(mode="json")
    assert dumped["feature_flags"] == flags
    assert RunRecord.model_validate(dumped).feature_flags == flags
