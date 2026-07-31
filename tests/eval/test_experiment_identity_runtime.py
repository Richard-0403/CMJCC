"""The experiment id must also cover the run inputs that are neither source nor config.

The defect pinned down here: ``experiment_id`` hashed the variants, the scenario ids, the
scenario content, the resolved config and the code fingerprint -- and nothing else. Three
things that decide what a run produces were invisible to it:

* the JOB CATALOG. It is a data file, so no source fingerprint moves when it is edited,
  and it is not part of the config; yet changing it changes every ranking and every
  relevance grade.
* the PROMPTS. Templates, not config values, so the same blind spot.
* the LLM BACKEND. ``JOBREC_LLM_MODEL`` and ``JOBREC_LLM_BASE_URL`` are read from the
  ENVIRONMENT by :class:`jobrec.llm.remote_provider.RemoteLLMProvider`, so a hybrid batch
  answered by one model and a hybrid batch answered by another shared one experiment id.
  The overwrite guard then reported the second as an idempotent re-run of the first --
  the exact classification that lets a new official run land on an older one.

The counter-requirement is asserted too: in ``deterministic`` mode no LLM is ever called,
so the backend variables must NOT enter the id. Otherwise the deterministic experiment id
would depend on the operator's shell, and the same batch would land under a different id on
a machine that happens to export those variables.

Secret safety (R26.1) is asserted structurally rather than by hoping: the endpoint is
reduced to its host before it is recorded, so a credential embedded in a base URL cannot
reach a manifest, a digest or a log line, and ``runtime_identity`` has no parameter that
could carry a key.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.domain.enums import RunMode
from jobrec.evaluation.experiment_identity import (
    RUNTIME_IDENTITY_FIELDS,
    endpoint_host,
    experiment_id,
    runtime_identity,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner
from jobrec.llm.remote_provider import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV

_INPUTS = {
    "variants": ["full"],
    "scenario_ids": ["SC-001"],
    "config_hash": "cfg-hash",
}

_IDENTITY = {
    "code_version": "0.1.0",
    "commit_hash": "a" * 40,
    "git_dirty": False,
    "source_fingerprint": "f" * 64,
    "execution_fingerprint": "e" * 64,
    "analysis_fingerprint": "d" * 64,
}

_RUNTIME = {
    "catalog_hash": "cat-1",
    "prompt_hash": "prm-1",
    "llm_mode": "hybrid",
    "llm_provider": "remote",
    "llm_model": "qwen-plus",
    "llm_endpoint": "api.example.com",
}


def _id(**runtime_overrides) -> str:
    return experiment_id(**_INPUTS, identity=_IDENTITY,
                         runtime={**_RUNTIME, **runtime_overrides})


# ------------------------------------------------- each runtime input moves the id
@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("catalog_hash", "cat-2"),
        ("prompt_hash", "prm-2"),
        ("llm_mode", "deterministic"),
        ("llm_provider", "mock"),
        ("llm_model", "gpt-4o-mini"),
        ("llm_endpoint", "api.other.com"),
    ],
)
def test_every_runtime_field_moves_the_id_independently(field: str, changed: str):
    """One field at a time, so a digest that silently drops one of them cannot pass."""
    assert _id() != _id(**{field: changed}), field


def test_a_rerun_with_the_same_runtime_is_still_idempotent():
    """Adding the runtime block must not break the intentional re-run case."""
    assert _id() == _id()


def test_adding_the_runtime_block_did_not_displace_the_older_inputs():
    """The code identity and the experiment inputs still move the id."""
    baseline = _id()
    assert experiment_id(**_INPUTS, identity={**_IDENTITY, "execution_fingerprint": "9" * 64},
                         runtime=_RUNTIME) != baseline
    assert experiment_id(**{**_INPUTS, "config_hash": "other"}, identity=_IDENTITY,
                         runtime=_RUNTIME) != baseline
    assert experiment_id(**_INPUTS, identity=_IDENTITY, runtime=_RUNTIME,
                         scenarios_fingerprint="s-1") != baseline


def test_an_absent_runtime_block_is_distinguishable_from_a_present_one():
    """``runtime=None`` is 'not recorded', not 'recorded as empty'."""
    assert experiment_id(**_INPUTS, identity=_IDENTITY, runtime=None) != _id()


# --------------------------------------------------------- the endpoint reduction
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/v1", "api.example.com"),
        ("https://API.Example.COM/v1", "api.example.com"),
        ("http://localhost:8000/v1", "localhost:8000"),
        ("localhost:8000", "localhost:8000"),
        ("api.example.com", "api.example.com"),
        ("https://api.example.com/v1?api-key=sk-secret", "api.example.com"),
        ("https://user:sk-secret@api.example.com/v1", "api.example.com"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_endpoint_host_keeps_only_what_identifies_the_backend(url, expected):
    assert endpoint_host(url) == expected


def test_two_proxies_on_one_host_are_two_backends():
    """The port is part of the identity: dropping it would merge distinct backends."""
    assert endpoint_host("http://localhost:8000/v1") != endpoint_host("http://localhost:9000/v1")


def test_a_credential_in_the_base_url_never_reaches_the_identity():
    """The endpoint is the one run input that can carry a key. It must be stripped."""
    secret = "sk-live-must-never-be-recorded"
    runtime = runtime_identity(
        catalog_hash="cat-1", prompt_hash="prm-1", llm_mode="hybrid", llm_provider="remote",
        llm_model="qwen-plus",
        llm_endpoint=f"https://svc:{secret}@api.example.com/v1?api-key={secret}",
    )
    assert secret not in json.dumps(runtime)
    assert runtime["llm_endpoint"] == "api.example.com"
    # And the same URL with a DIFFERENT key is the same experiment: rotating a credential
    # is not a new experiment, so it must not fork the id.
    rotated = runtime_identity(
        catalog_hash="cat-1", prompt_hash="prm-1", llm_mode="hybrid", llm_provider="remote",
        llm_model="qwen-plus", llm_endpoint="https://svc:other-key@api.example.com/v1",
    )
    assert rotated == runtime


def test_runtime_identity_cannot_be_handed_an_api_key():
    """Structural guard: no parameter exists that a key could be passed through.

    Asserted on the signature rather than on behaviour, because the failure mode is a
    future edit ADDING such a parameter -- which no output-based test would catch.
    """
    params = set(inspect.signature(runtime_identity).parameters)
    assert params == set(RUNTIME_IDENTITY_FIELDS)
    assert not any("key" in name or "secret" in name or "token" in name for name in params)


def test_runtime_identity_records_exactly_the_declared_fields():
    runtime = runtime_identity(catalog_hash="c", prompt_hash="p", llm_mode="deterministic",
                              llm_provider="mock")
    assert list(runtime) == list(RUNTIME_IDENTITY_FIELDS)
    assert runtime["llm_model"] is None and runtime["llm_endpoint"] is None


# ------------------------------------------- what the runner actually reports
def _runner(tmp_path: Path, mode: RunMode, provider: str = "remote") -> ExperimentRunner:
    """A runner over a one-line scenario file; nothing is executed, only inspected."""
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(json.dumps({
        "scenario_id": "SC-RT-01",
        "scenario_type": "basic",
        "profile": {"candidate_id": "SC-RT-01-cand", "skills": ["Python"],
                    "years_experience": 3},
        "turns": ["I want a data analyst role in Kuala Lumpur."],
        "expects": {"response_type": "recommendation"},
    }) + "\n", encoding="utf-8")
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.llm.mode = mode
    config.llm.provider = provider
    return ExperimentRunner(config, "data/processed/jobs.jsonl", str(scenarios),
                            out_dir=str(tmp_path / "runs"))


def test_deterministic_mode_ignores_the_backend_environment(tmp_path: Path, monkeypatch):
    """No LLM is called, so the shell must not be able to fork the deterministic id.

    Without this the deterministic re-run would be irreproducible on any machine that
    exports the hybrid credentials, which is every machine that can run the hybrid arm.
    """
    monkeypatch.setenv(MODEL_ENV, "some-other-model")
    monkeypatch.setenv(BASE_URL_ENV, "https://elsewhere.example.com/v1")
    runtime = _runner(tmp_path, RunMode.DETERMINISTIC)._runtime_identity("cat-1")

    assert runtime["llm_mode"] == "deterministic"
    assert runtime["llm_model"] is None
    assert runtime["llm_endpoint"] is None
    assert "elsewhere.example.com" not in json.dumps(runtime)


def test_hybrid_with_a_mock_provider_names_no_backend(tmp_path: Path, monkeypatch):
    """``mode: hybrid`` + ``provider: mock`` contacts nothing, so it must name nothing.

    Found by a mock-backed hybrid smoke: the identity had gated on the MODE alone, so this
    run was stamped with the environment's model and endpoint even though every call was
    answered by the deterministic mock. That is provenance naming a backend that never
    answered, and it also let an unrelated exported variable move the experiment id.
    """
    monkeypatch.setenv(MODEL_ENV, "qwen-plus")
    monkeypatch.setenv(BASE_URL_ENV, "https://dashscope.example.com/compatible-mode/v1")
    runtime = _runner(tmp_path, RunMode.HYBRID, provider="mock")._runtime_identity("cat-1")

    assert runtime["llm_mode"] == "hybrid"
    assert runtime["llm_provider"] == "mock"
    assert runtime["llm_model"] is None
    assert runtime["llm_endpoint"] is None
    assert "dashscope.example.com" not in json.dumps(runtime)


def test_the_identity_and_the_provider_factory_agree_on_what_is_remote(tmp_path: Path):
    """One predicate decides both, so the recorded backend cannot drift from the real one."""
    from jobrec.orchestration.orchestrator import uses_remote_backend

    for mode, provider, expected in [
        (RunMode.DETERMINISTIC, "mock", False),
        (RunMode.DETERMINISTIC, "remote", False),
        (RunMode.HYBRID, "mock", False),
        (RunMode.HYBRID, "remote", True),
    ]:
        runner = _runner(tmp_path, mode, provider=provider)
        assert uses_remote_backend(runner.config) is expected, (mode, provider)
        named = runner._runtime_identity("cat-1")["llm_model"] is not None
        assert named is expected, (mode, provider)


def test_hybrid_mode_records_the_backend_named_by_the_environment(tmp_path: Path,
                                                                 monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "qwen-plus")
    monkeypatch.setenv(BASE_URL_ENV, "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv(API_KEY_ENV, "sk-live-must-never-be-recorded")
    runtime = _runner(tmp_path, RunMode.HYBRID,
                      provider="remote")._runtime_identity("cat-1")

    assert runtime["llm_mode"] == "hybrid"
    assert runtime["llm_model"] == "qwen-plus"
    assert runtime["llm_endpoint"] == "dashscope.example.com"
    assert "sk-live-must-never-be-recorded" not in json.dumps(runtime)
    # The prompts and the catalog are reported, so both can move the id.
    assert runtime["catalog_hash"] == "cat-1"
    assert runtime["prompt_hash"]


def test_switching_the_model_gives_the_hybrid_batch_a_new_id(tmp_path: Path, monkeypatch):
    """End to end over the runner: the case the overwrite guard used to misread."""
    monkeypatch.setenv(BASE_URL_ENV, "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv(MODEL_ENV, "qwen-plus")
    first = _runner(tmp_path, RunMode.HYBRID)._runtime_identity("cat-1")
    monkeypatch.setenv(MODEL_ENV, "qwen-max")
    second = _runner(tmp_path, RunMode.HYBRID)._runtime_identity("cat-1")

    assert experiment_id(**_INPUTS, identity=_IDENTITY, runtime=first) != experiment_id(
        **_INPUTS, identity=_IDENTITY, runtime=second)
