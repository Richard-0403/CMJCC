"""P0-6a: the three provenance gaps that survived the first identity pass.

1. Session ids were ``random_id("sess")`` even inside an experiment. ``run_id`` is
   ``content_id("run", session_id, version, text)``, so a second execution of the SAME
   frozen inputs produced a different id for every run: the two batches could not be diffed
   run by run, and the idempotence claim held for the directory name while being false for
   everything inside it. Determinism is scoped to ``ExperimentRunner`` -- live sessions must
   stay random, because two real conversations must never share an id.

2. Model-call records described only what was REQUESTED. The server's own answer to "what
   answered this" -- the completion id, the ``system_fingerprint``, and the ``model`` the
   server says it used, which an alias or a gateway can make different from the one asked
   for -- was discarded, so "same model" was an assumption rather than a record.

3. The endpoint was reduced to its host. Safe, but too lossy: OpenAI-compatible deployments
   are routinely distinguished by path alone (``/v1`` versus ``/compatible-mode/v1``), so two
   different backends collided on one identity and the experiment id could not separate them.

The security requirement is unchanged and asserted directly: no credential may reach an
identity, a manifest or a digest, whether it arrives as userinfo or as a query token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import RunMode
from jobrec.evaluation.experiment_identity import (
    EXPERIMENT_MANIFEST_FILENAME,
    endpoint_identity,
    experiment_id,
    runtime_identity,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"

_SCENARIO = {
    "scenario_id": "SC-PROV-01",
    "scenario_type": "basic",
    "profile": {"candidate_id": "SC-PROV-01-cand", "skills": ["Python"],
                "years_experience": 3},
    "turns": ["I want a data analyst role in Kuala Lumpur, at least RM4000."],
    "expects": {"response_type": "recommendation"},
}


# ------------------------------------------------------- 1. deterministic session ids
def _tiny_inputs(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in Path(CATALOG).read_text().splitlines()
            if line.strip()]
    tiny = [j for j in jobs if j["city"] == "Kuala Lumpur"][:6]
    catalog = root / "jobs.jsonl"
    catalog.write_text("\n".join(json.dumps(j) for j in tiny) + "\n", encoding="utf-8")
    scenarios = root / "scenarios.jsonl"
    scenarios.write_text(json.dumps(_SCENARIO) + "\n", encoding="utf-8")
    return str(catalog), str(scenarios)


def _runner(inputs: tuple[str, str], out_dir: Path) -> ExperimentRunner:
    config = load_config(CONFIG, base_dir="configs")
    config.experiment.repeat_count = 1
    return ExperimentRunner(config, inputs[0], inputs[1], out_dir=str(out_dir))


def test_a_live_session_is_still_random(tmp_path: Path):
    """Only the runner is deterministic. Two real conversations must not collide."""
    service = AppService(load_config(CONFIG, base_dir="configs"), CATALOG)
    candidate = service.create_candidate({"candidate_id": "live-cand", "skills": []})
    first = service.create_session(candidate.candidate_id, "full")
    second = service.create_session(candidate.candidate_id, "full")

    assert first != second
    assert first.startswith("sess-")


def test_a_caller_may_supply_a_session_id(tmp_path: Path):
    service = AppService(load_config(CONFIG, base_dir="configs"), CATALOG)
    candidate = service.create_candidate({"candidate_id": "given-cand", "skills": []})
    assert service.create_session(candidate.candidate_id, "full",
                                 session_id="sess-chosen") == "sess-chosen"


def test_two_runs_of_the_same_frozen_inputs_reproduce_every_id(tmp_path: Path):
    """The point of the change: a re-run is diffable run by run, not just directory by
    directory."""
    inputs = _tiny_inputs(tmp_path / "in")
    first = _runner(inputs, tmp_path / "a").run(["full"])
    second = _runner(inputs, tmp_path / "b").run(["full"])

    assert first["experiment_id"] == second["experiment_id"]

    def ids(manifest) -> list[tuple]:
        exp_dir = Path(manifest["experiment_dir"])
        out = []
        for rel in manifest["run_manifests"]:
            run_dir = exp_dir / Path(rel).parent
            record = json.loads((run_dir / "run_record.json").read_text())
            dialogue = json.loads((run_dir / "dialogue_state.json").read_text())
            out.append((record["run_id"], dialogue["session_id"],
                        [t["turn_id"] for t in dialogue["turns"]],
                        dialogue["active_search_id"]))
        return out

    assert ids(first) == ids(second), "a re-run of frozen inputs changed its run-level ids"


def test_session_ids_do_not_collide_across_variants_scenarios_or_repeats(tmp_path: Path):
    """Every coordinate of the derivation is load-bearing."""
    derive = ExperimentRunner._session_id
    exp = tmp_path / "exp-abc123abc123"
    base = derive(exp, "full", "SC-1", 0, 0)
    others = [
        derive(exp, "no_memory", "SC-1", 0, 0),
        derive(exp, "full", "SC-2", 0, 0),
        derive(exp, "full", "SC-1", 1, 0),
        derive(exp, "full", "SC-1", 0, 1),
        derive(tmp_path / "exp-999999999999", "full", "SC-1", 0, 0),
    ]
    assert len(set([base, *others])) == len(others) + 1, (base, others)
    assert derive(exp, "full", "SC-1", 0, 0) == base


def test_changing_a_session_id_does_not_change_the_experiment_id(tmp_path: Path):
    """The session id must stay out of the experiment digest, or the derivation is circular.

    The experiment id is computed before any session exists; it is a function of the inputs,
    the code and the runtime backend. This pins the direction of the dependency.
    """
    inputs = _tiny_inputs(tmp_path / "in")
    manifest = _runner(inputs, tmp_path / "runs").run(["full"])
    on_disk = json.loads(
        (Path(manifest["experiment_dir"]) / EXPERIMENT_MANIFEST_FILENAME).read_text())

    assert "session_id" not in json.dumps(on_disk["runtime_identity"])
    # Re-deriving the id from the recorded inputs alone reproduces it, so nothing
    # session-scoped contributed to it.
    assert experiment_id(
        variants=on_disk["variants"], scenario_ids=[_SCENARIO["scenario_id"]],
        config_hash=on_disk["config_hash"],
        identity={k: on_disk[k] for k in ("code_version", "execution_fingerprint")},
        scenarios_fingerprint=on_disk["scenarios_hash"],
        runtime=on_disk["runtime_identity"],
    ) == manifest["experiment_id"]


# --------------------------------------------------------- 2. response provenance
def test_the_servers_own_answer_about_what_replied_is_recorded():
    from jobrec.llm.remote_provider import _response_provenance

    recorded = _response_provenance({
        "id": "chatcmpl-abc123",
        "system_fingerprint": "fp_44709d6fcb",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"finish_reason": "stop"}],
    })
    assert recorded == {"response_id": "chatcmpl-abc123",
                        "system_fingerprint": "fp_44709d6fcb",
                        "response_model": "gpt-4o-mini-2024-07-18"}


def test_a_field_the_server_omitted_is_absent_rather_than_invented():
    """Many OpenAI-compatible proxies omit ``system_fingerprint``. A fabricated value would
    be indistinguishable from a real one when reading the bundle."""
    from jobrec.llm.remote_provider import _response_provenance

    recorded = _response_provenance({"id": "chatcmpl-1", "choices": []})
    assert recorded == {"response_id": "chatcmpl-1"}
    assert "system_fingerprint" not in recorded
    assert "response_model" not in recorded
    # Non-string and empty values are not provenance either.
    assert _response_provenance({"id": "", "system_fingerprint": 42, "model": None}) == {}
    assert _response_provenance("not a dict") == {}


def test_the_recorded_model_may_differ_from_the_requested_one():
    """An alias resolves to a dated build and a gateway may route elsewhere; that is the
    reason to record what the SERVER said rather than what was asked for."""
    from jobrec.llm.remote_provider import _response_provenance

    recorded = _response_provenance({"model": "qwen-plus-2025-01-25"})
    assert recorded["response_model"] == "qwen-plus-2025-01-25"


def test_response_provenance_carries_no_prompt_or_credential():
    from jobrec.llm.remote_provider import _response_provenance

    recorded = _response_provenance({
        "id": "chatcmpl-1", "model": "m",
        # A server echoing the request must not widen what gets stored.
        "prompt": "the candidate said ...", "api_key": "sk-must-never-be-recorded",
    })
    assert set(recorded) == {"response_id", "response_model"}
    assert "sk-must-never-be-recorded" not in json.dumps(recorded)


# ------------------------------------------------------------- 3. endpoint identity
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/v1", "https://api.example.com/v1"),
        ("https://API.Example.COM/v1", "https://api.example.com/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("https://api.example.com/compatible-mode/v1",
         "https://api.example.com/compatible-mode/v1"),
        # Normalisation: one endpoint, not three.
        ("https://api.example.com/v1/", "https://api.example.com/v1"),
        ("https://api.example.com//v1", "https://api.example.com/v1"),
        ("https://api.example.com", "https://api.example.com"),
        ("api.example.com", "https://api.example.com"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_endpoint_identity_keeps_the_distinguishing_parts(url, expected):
    assert endpoint_identity(url) == expected


def test_two_deployments_on_one_host_are_two_backends():
    """The collision the host-only reduction caused."""
    a = endpoint_identity("https://api.example.com/v1")
    b = endpoint_identity("https://api.example.com/compatible-mode/v1")
    assert a != b

    inputs = {"variants": ["full"], "scenario_ids": ["SC-1"], "config_hash": "c"}
    ident = {"code_version": "0.1.0", "execution_fingerprint": "e" * 64}

    def id_for(endpoint: str) -> str:
        return experiment_id(**inputs, identity=ident, runtime=runtime_identity(
            catalog_hash="cat", prompt_hash="prm", llm_mode="hybrid",
            llm_provider="remote", llm_model="m", llm_endpoint=endpoint))

    assert id_for("https://api.example.com/v1") != id_for(
        "https://api.example.com/compatible-mode/v1")
    assert id_for("http://localhost:8000/v1") != id_for("http://localhost:9000/v1")
    assert id_for("http://api.example.com/v1") != id_for("https://api.example.com/v1")


@pytest.mark.parametrize(
    "leaky",
    [
        "https://user:sk-live-secret@api.example.com/v1",
        "https://sk-live-secret@api.example.com/v1",
        "https://api.example.com/v1?api-key=sk-live-secret",
        "https://api.example.com/v1?token=sk-live-secret&x=1",
        "https://api.example.com/v1#sk-live-secret",
        "https://user:sk-live-secret@api.example.com/v1?api-key=sk-live-secret",
    ],
)
def test_no_credential_survives_into_the_identity(leaky: str):
    """Userinfo, query and fragment are all removed, and the result is rebuilt from parsed
    structure rather than sliced out of the input."""
    identity = endpoint_identity(leaky)
    assert identity == "https://api.example.com/v1", identity
    assert "sk-live-secret" not in identity

    runtime = runtime_identity(catalog_hash="cat", prompt_hash="prm", llm_mode="hybrid",
                              llm_provider="remote", llm_model="m", llm_endpoint=leaky)
    assert "sk-live-secret" not in json.dumps(runtime)


def test_rotating_a_credential_is_not_a_new_experiment():
    """Same endpoint, different key -> same id. Otherwise a key rotation forks the run."""
    ident = {"code_version": "0.1.0", "execution_fingerprint": "e" * 64}
    inputs = {"variants": ["full"], "scenario_ids": ["SC-1"], "config_hash": "c"}

    def id_for(endpoint: str) -> str:
        return experiment_id(**inputs, identity=ident, runtime=runtime_identity(
            catalog_hash="cat", prompt_hash="prm", llm_mode="hybrid",
            llm_provider="remote", llm_model="m", llm_endpoint=endpoint))

    assert id_for("https://u:key-one@api.example.com/v1") == id_for(
        "https://u:key-two@api.example.com/v1")
    assert id_for("https://api.example.com/v1?api-key=one") == id_for(
        "https://api.example.com/v1?api-key=two")


def test_the_runner_records_the_full_endpoint_identity(tmp_path: Path, monkeypatch):
    from jobrec.llm.remote_provider import BASE_URL_ENV, MODEL_ENV

    monkeypatch.setenv(BASE_URL_ENV, "https://svc:secret@gw.example.com/deploy-a/v1?k=z")
    monkeypatch.setenv(MODEL_ENV, "qwen-plus")
    config = load_config(CONFIG, base_dir="configs")
    config.llm.mode = RunMode.HYBRID
    config.llm.provider = "remote"
    scenarios = tmp_path / "s.jsonl"
    scenarios.write_text(json.dumps(_SCENARIO) + "\n", encoding="utf-8")
    runner = ExperimentRunner(config, CATALOG, str(scenarios), out_dir=str(tmp_path / "r"))

    runtime = runner._runtime_identity("cat-1")
    assert runtime["llm_endpoint"] == "https://gw.example.com/deploy-a/v1"
    assert "secret" not in json.dumps(runtime)
