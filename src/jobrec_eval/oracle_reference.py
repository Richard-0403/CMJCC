"""The CANONICAL scenario-level reference the relevance oracle grades against.

Why this module exists
----------------------
The oracle needs, per scenario, an authoritative constraint bundle (which jobs are
hard-eligible) plus the stated search intent (target roles, skills). Those used to be
lifted out of the experiment's own run bundles: :func:`jobrec_eval.relevance.build_references`
kept the FIRST ``full``-variant bundle it happened to encounter, i.e. repeat 0 of the
condition under evaluation. Three problems followed from that, all of which reach the
reported numbers:

1. **Construct validity.** The labels a system is scored against were produced by that
   same system's best-equipped condition. A variant was effectively graded on how
   closely it reproduced ``full``.
2. **Sampling artefact.** With repeats > 1 the retained bundle is one arbitrary draw. In
   the hybrid experiment repeat 0 is the MINORITY outcome on two scenarios, so the whole
   label universe -- and every ranking, task-success and significance figure downstream
   -- turned on which repeat happened to be enumerated first.
3. **Cross-experiment coupling.** The deterministic and hybrid experiments each grew
   their own labels, so their ranking numbers were not measured on a common yardstick.

The fix is a reference that is a pure function of the EXPERIMENT INPUTS -- the scenario
file and the catalog -- computed once under one fixed, model-free condition, then frozen
on disk with its own fingerprint and reused by every experiment that shares those
inputs. Two rejected alternatives, for the record: a majority vote across repeats (still
system-derived, and undefined at 1 repeat), and failing loudly when repeats disagree
(which would leave the hybrid experiment unanalysable).

What is still a threat
----------------------
This is NOT human annotation. The reference is derived by the system's own deterministic
rule extraction and constraint semantics, so the oracle remains a transparent proxy
rather than ground truth, and the report keeps declaring that. What the canonical
reference removes is narrower but real: variant privilege, repeat luck, and the two
experiments disagreeing about what the scenario asked for.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobrec.catalog import catalog_hash, load_catalog
from jobrec.config import AppConfig
from jobrec.domain.enums import RunMode
from jobrec.utils.hashing import sha256_of_bytes, stable_hash

#: Bumped whenever the DERIVATION of the canonical reference changes (not when a
#: scenario or the catalog changes -- those move the input fingerprint instead). A
#: frozen artifact carrying a different version is rejected rather than reused, so a
#: derivation change can never be masked by a stale file.
CANONICAL_ORACLE_VERSION = "2.0.0"

#: Name under which the artifact is republished into an analysis directory.
CANONICAL_ORACLE_FILENAME = "canonical_oracle.json"

#: Prefix of the frozen artifact, written beside the scenario file because it is an INPUT
#: to evaluation, not an output of one particular experiment. The scenario file's stem is
#: appended (see :func:`frozen_artifact_path`): a directory holds several scenario files
#: -- the 42-scenario set and the 12-scenario subset live side by side -- and one shared
#: filename would make them overwrite each other's reference, so whichever ran second
#: would abort on a stale-oracle error.
CANONICAL_ORACLE_PREFIX = "canonical_oracle"

#: The single condition the canonical reference is derived under. ``full`` because it is
#: the only variant with every information source enabled -- a reference built under a
#: memory-less or context-less condition would encode the ablation's blind spots as
#: ground truth. ``deterministic`` because a model sample must never move the labels:
#: this is what lets the deterministic and hybrid experiments share one yardstick.
CANONICAL_VARIANT = "full"
CANONICAL_REPEATS = 1


class StaleCanonicalOracleError(RuntimeError):
    """A frozen canonical oracle does not match the inputs it is about to grade.

    Raised instead of silently regenerating: a reference computed over a different
    scenario file or catalog would produce labels that look authoritative while grading
    jobs the experiment never saw.
    """


@dataclass(frozen=True)
class CanonicalReferences:
    """The frozen reference plus everything needed to audit where it came from."""

    #: ``{scenario_id: {"job_context": ..., "active_search": ...}}`` -- the exact shape
    #: :func:`jobrec_eval.relevance.grade_catalog` consumes, so it is a drop-in
    #: replacement for the old bundle-derived dict.
    references: dict[str, dict]
    #: Fingerprint of the INPUTS (scenarios + catalog + derivation version). Two runs
    #: with the same value must see the same labels.
    inputs_fingerprint: str
    #: Fingerprint of the LABEL-DETERMINING projection of the reference
    #: (:func:`grading_projection`), not of the raw payload: the raw payload carries
    #: ``normalized_at`` and timestamp-derived ids, so a hash over it would differ
    #: between two identical derivations and could prove nothing about the labels.
    reference_fingerprint: str
    provenance: dict[str, Any]

    def as_artifact(self) -> dict[str, Any]:
        """The on-disk form of the frozen artifact."""
        return {
            "canonical_oracle_version": CANONICAL_ORACLE_VERSION,
            "inputs_fingerprint": self.inputs_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "provenance": dict(self.provenance),
            "references": self.references,
        }


#: The parts of ``active_search`` that actually reach a grade
#: (:func:`jobrec_eval.relevance._role_score` and ``_skill_coverage``).
_GRADING_ACTIVE_KEYS = ("target_roles", "skills_have")

#: The parts of a ``ConstraintDefinition`` that decide eligibility. ``constraint_id`` and
#: ``evidence_ids`` are excluded on purpose: they are content ids over objects carrying
#: wall-clock stamps, so including them would make the fingerprint change on every
#: derivation and thereby prove nothing.
_GRADING_CONSTRAINT_KEYS = ("field_name", "operator", "expected_value", "strength",
                            "weight", "unknown_policy", "rule_id")


def grading_projection(references: dict[str, dict]) -> dict[str, Any]:
    """The label-determining content of a reference set, stripped of volatile stamps.

    Two reference sets with the same projection MUST produce the same relevance labels,
    which is the only property a fingerprint over the reference is worth asserting. The
    raw reference carries ``normalized_at`` and content ids derived from timestamped
    objects, so a fingerprint over it would differ between two identical derivations.
    """
    out: dict[str, Any] = {}
    for scenario_id in sorted(references):
        ref = references[scenario_id] or {}
        context = ref.get("job_context") or {}
        active = ref.get("active_search") or {}
        constraints = [
            {key: constraint.get(key) for key in _GRADING_CONSTRAINT_KEYS}
            for constraint in (context.get("constraints") or [])
        ]
        out[scenario_id] = {
            # Sorted by their serialized form: constraint ORDER never affects a grade.
            "constraints": sorted(
                json.dumps(c, sort_keys=True, default=str) for c in constraints),
            **{key: sorted(str(v) for v in (active.get(key) or []))
               for key in _GRADING_ACTIVE_KEYS},
        }
    return out


def frozen_artifact_path(scenarios_path: str | Path) -> Path:
    """Where ``scenarios_path``'s frozen reference lives: beside it, keyed by its stem."""
    p = Path(scenarios_path)
    return p.parent / f"{CANONICAL_ORACLE_PREFIX}_{p.stem}.json"


def _file_fingerprint(path: str | Path) -> str:
    return sha256_of_bytes(Path(path).read_bytes())


def inputs_fingerprint(scenarios_path: str | Path, catalog_path: str | Path) -> str:
    """Fingerprint of everything the canonical reference is a function of.

    The catalog contributes its NORMALIZED hash rather than its file bytes, so a
    re-serialised but semantically identical catalog does not invalidate the artifact,
    while an actual job change does.
    """
    return stable_hash({
        "canonical_oracle_version": CANONICAL_ORACLE_VERSION,
        "scenarios_sha256": _file_fingerprint(scenarios_path),
        "catalog_hash": catalog_hash(load_catalog(catalog_path)),
        "variant": CANONICAL_VARIANT,
        "repeats": CANONICAL_REPEATS,
        "llm_mode": RunMode.DETERMINISTIC.value,
    })


def canonical_config(config: AppConfig) -> AppConfig:
    """``config`` reduced to the canonical condition.

    Only the knobs that would let a model sample or a repeat count leak into the labels
    are pinned; everything else (constraint semantics, top_k, dialogue budget) is left
    as the experiment configured it, because the reference has to describe the same
    world the experiment ran in.
    """
    pinned = config.model_copy(deep=True)
    pinned.llm.mode = RunMode.DETERMINISTIC
    pinned.experiment.repeat_count = CANONICAL_REPEATS
    return pinned


def build_canonical_references(
    scenarios_path: str | Path,
    catalog_path: str | Path,
    config: AppConfig,
    *,
    work_dir: str | Path | None = None,
) -> CanonicalReferences:
    """Derive the canonical reference for every scenario in ``scenarios_path``.

    Each scenario is driven once through the real pipeline under
    :func:`canonical_config`, including the clarification loop -- a clarification
    scenario states no role until the loop answers for one, and a reference without a
    role grades the entire catalog 0, which would make that scenario's ranking metrics
    undefined rather than merely different.

    The pass writes into a throwaway directory: the durable output is the reference
    artifact, not another copy of the run bundles.
    """
    # Imported here: this is the only place jobrec_eval drives the runner directly, and
    # a module-level import would make every metrics import pull in the whole pipeline.
    from jobrec.evaluation.experiment_runner import ExperimentRunner

    pinned = canonical_config(config)
    scenarios = _read_scenario_records(scenarios_path)

    references: dict[str, dict] = {}
    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cmjcc-canonical-oracle-") as tmp:
        root = Path(work_dir) if work_dir is not None else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        runner = ExperimentRunner(
            pinned, str(catalog_path), str(scenarios_path), out_dir=str(root / "_runs"))
        for scenario in scenarios:
            row, _failure = runner._run_one(
                CANONICAL_VARIANT, scenario, 0, root / "canonical")
            run_dir = Path(row["run_dir"])
            job_context = _read_json(run_dir / "job_context_state.json")
            active_search = _read_json(run_dir / "active_search_state.json")
            if not job_context or not active_search:
                # Recorded, not silently skipped: a scenario with no reference gets no
                # labels, so its ranking metrics vanish -- that has to be visible.
                missing.append(scenario["scenario_id"])
                continue
            references[scenario["scenario_id"]] = {
                "job_context": job_context, "active_search": active_search}

    provenance = {
        "derivation": "canonical_pass",
        "variant": CANONICAL_VARIANT,
        "repeats": CANONICAL_REPEATS,
        "llm_mode": RunMode.DETERMINISTIC.value,
        "scenarios_path": str(scenarios_path),
        "scenarios_sha256": _file_fingerprint(scenarios_path),
        "catalog_path": str(catalog_path),
        "catalog_hash": catalog_hash(load_catalog(catalog_path)),
        "scenario_count": len(scenarios),
        "referenced_scenario_count": len(references),
        "scenarios_without_reference": sorted(missing),
    }
    return CanonicalReferences(
        references=references,
        inputs_fingerprint=inputs_fingerprint(scenarios_path, catalog_path),
        reference_fingerprint=stable_hash(grading_projection(references)),
        provenance=provenance,
    )


def load_frozen_references(path: str | Path) -> CanonicalReferences | None:
    """Load a frozen artifact, or ``None`` when there is none to load."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return CanonicalReferences(
        references=data.get("references") or {},
        inputs_fingerprint=str(data.get("inputs_fingerprint") or ""),
        reference_fingerprint=str(data.get("reference_fingerprint") or ""),
        provenance={**(data.get("provenance") or {}),
                    "loaded_from": str(p),
                    "canonical_oracle_version":
                        data.get("canonical_oracle_version")},
    )


def load_or_build_canonical_references(
    scenarios_path: str | Path,
    catalog_path: str | Path,
    config: AppConfig,
    *,
    frozen_path: str | Path | None = None,
    freeze: bool = True,
) -> CanonicalReferences:
    """Reuse the frozen canonical reference, or derive and freeze it on first use.

    Reuse is what makes the reference a shared yardstick: the deterministic and hybrid
    experiments read the SAME file, so their ranking numbers are comparable. A frozen
    artifact whose ``inputs_fingerprint`` does not match the current scenario file and
    catalog raises :class:`StaleCanonicalOracleError` -- regenerating silently would
    change the labels under a heading claiming they were frozen, and reusing it silently
    would grade jobs the experiment never saw.
    """
    target = (Path(frozen_path) if frozen_path is not None
              else frozen_artifact_path(scenarios_path))
    expected = inputs_fingerprint(scenarios_path, catalog_path)

    frozen = load_frozen_references(target)
    if frozen is not None:
        if frozen.inputs_fingerprint != expected:
            raise StaleCanonicalOracleError(
                f"the frozen canonical oracle at {target} was derived for a different "
                f"scenario file / catalog / derivation version "
                f"(artifact {frozen.inputs_fingerprint or 'missing'!r} != current "
                f"{expected!r}). Delete it to re-derive, and re-report every "
                f"grade-derived metric: the labels will change."
            )
        return frozen

    built = build_canonical_references(scenarios_path, catalog_path, config)
    if freeze:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(built.as_artifact(), indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        built = CanonicalReferences(
            references=built.references,
            inputs_fingerprint=built.inputs_fingerprint,
            reference_fingerprint=built.reference_fingerprint,
            provenance={**built.provenance, "frozen_to": str(target)},
        )
    return built


def _read_scenario_records(path: str | Path) -> list[dict]:
    """The raw scenario dicts, in file order (the runner consumes this shape)."""
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
