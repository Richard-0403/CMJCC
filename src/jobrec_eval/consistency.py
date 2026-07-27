"""Pre-comparison configuration-consistency gate (R15 + R32.7).

Before any comparison report is produced, the runs being compared must share the
inputs and settings that make the comparison meaningful: the same catalog, the
same scenario set, the same prompts, the same model settings, the same top-k,
retrieval pool size and seed, and the same source commit (R15.1). For an ablation
pair the gate additionally requires that ONLY the target mechanism's feature
flags differ, so that any measured delta is attributable to that single mechanism
(R32.7).

The two public entry points work on the ``run_manifest.json`` payloads produced
by :func:`jobrec.evaluation.manifest.build_run_manifest`:

* :func:`check_consistency` -- pure verification returning a
  :class:`ConsistencyReport`; it also stamps the resulting consistency flags into
  each inspected manifest (R15.3).
* :func:`require_consistent` -- the gate itself: raises
  :class:`ConsistencyError` when the compared runs do not match, so a caller
  (``report.py``) can stop before generating output (R15.2).

Field sourcing is deliberately tolerant: each logical field is looked up through
an ordered list of candidate paths, so the checker keeps working as the manifest
grows richer. A field that no manifest records is reported as *unavailable*
rather than silently treated as matching; a field recorded by some runs but not
others is a mismatch, because partial provenance cannot be verified.

Manifest coverage note: the current run manifest records the catalog and prompt
hashes, the commit hash, the resolved feature flags and a non-sensitive
provider/model/mode summary. The scenario hash lives in the experiment-level
manifest, and top-k / retrieval pool size / seed / sampling temperatures are
folded into ``config_hash`` rather than exposed as separate fields. Those fields
are therefore checked when present and otherwise listed in
``unavailable_fields``; ``config_hash`` is compared *within* each variant (it
necessarily differs across variants, since the variant is part of the config) so
the settings it encodes are still verified for every repeat of a variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any

from jobrec.domain.enums import ExperimentVariant
from jobrec.orchestration.feature_flags import FeatureFlags, flag_diff

__all__ = [
    "ConsistencyError",
    "ConsistencyReport",
    "FieldFinding",
    "FlagFinding",
    "check_consistency",
    "load_run_manifests",
    "require_consistent",
    "save_run_manifests",
]

#: Sentinel for "this manifest does not record the field at all".
_MISSING = object()

#: Key under which :func:`load_run_manifests` remembers a manifest's file path.
#: Private (leading underscore) keys are stripped again before writing to disk.
_SOURCE_KEY = "_source_path"

#: Logical field -> ordered candidate dotted paths inside a run manifest.
#: Every field of R15.1 that a run manifest can express is listed here.
_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "catalog_hash": ("hashes.catalog_hash", "catalog_hash"),
    "scenario_hash": (
        "hashes.scenario_hash",
        "hashes.scenarios_hash",
        "scenario_hash",
        "scenarios_hash",
    ),
    "prompt_hash": ("hashes.prompt_hash", "prompt_hash"),
    "model_settings": ("model_settings", "settings.model_settings", "api_summary"),
    "top_k": ("settings.top_k", "experiment.top_k", "top_k"),
    "pool_size": (
        "settings.retrieval_pool_size",
        "experiment.retrieval_pool_size",
        "settings.pool_size",
        "retrieval_pool_size",
        "pool_size",
    ),
    "seed": ("settings.random_seed", "experiment.random_seed", "random_seed", "seed"),
    "commit_hash": ("commit_hash", "commit"),
}

#: Fields that legitimately differ ACROSS variants and are therefore compared
#: only among runs of the same variant. ``config_hash`` covers top-k, pool size,
#: seed, ranking weights and sampling temperatures.
_VARIANT_SCOPED_PATHS: dict[str, tuple[str, ...]] = {
    "config_hash": ("hashes.config_hash", "config_hash"),
}


# --------------------------------------------------------------------- findings
@dataclass(frozen=True)
class FieldFinding:
    """Outcome of comparing one logical field across the compared runs."""

    field: str
    consistent: bool
    available: bool
    values: dict[str, Any]
    missing_in: tuple[str, ...] = ()
    scope: str = "all"

    def describe(self) -> str:
        if not self.available:
            return f"{self.field}: not recorded by any compared run"
        detail = ", ".join(f"{label}={value!r}" for label, value in sorted(self.values.items()))
        if self.missing_in:
            detail += f"; missing in {', '.join(self.missing_in)}"
        scope = "" if self.scope == "all" else f" (per {self.scope})"
        return f"{self.field}{scope}: {detail}"


@dataclass(frozen=True)
class FlagFinding:
    """Outcome of comparing the resolved feature flags of one pair of runs."""

    left: str
    right: str
    differing_flags: tuple[str, ...]
    consistent: bool
    reason: str

    def describe(self) -> str:
        diff = ", ".join(self.differing_flags) or "<none>"
        return f"{self.left} vs {self.right}: differing flags [{diff}] -- {self.reason}"


@dataclass(frozen=True)
class ConsistencyReport:
    """Verification result for a set of run manifests."""

    consistent: bool
    run_labels: tuple[str, ...]
    field_findings: tuple[FieldFinding, ...]
    flag_findings: tuple[FlagFinding, ...]
    flags: dict[str, bool]
    mismatched_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    target_flag_set: tuple[str, ...] | None = None

    def summary(self) -> str:
        """Human-readable summary, used verbatim in the gate's error message."""
        head = (
            f"configuration consistency {'OK' if self.consistent else 'FAILED'} "
            f"across {len(self.run_labels)} run(s)"
        )
        lines = [head]
        for finding in self.field_findings:
            if not finding.consistent:
                lines.append(f"  mismatch: {finding.describe()}")
        for flag_finding in self.flag_findings:
            if not flag_finding.consistent:
                lines.append(f"  mismatch: {flag_finding.describe()}")
        if self.unavailable_fields:
            lines.append(f"  not recorded (unverified): {', '.join(self.unavailable_fields)}")
        return "\n".join(lines)


class ConsistencyError(RuntimeError):
    """Raised by :func:`require_consistent` when compared runs do not match."""

    def __init__(self, report: ConsistencyReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass(frozen=True)
class _Run:
    label: str
    variant: str | None
    manifest: dict[str, Any]


# ------------------------------------------------------------------- primitives
def _lookup(manifest: dict[str, Any], paths: tuple[str, ...]) -> Any:
    """Return the first value found at any of ``paths``, else :data:`_MISSING`."""
    for path in paths:
        node: Any = manifest
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = _MISSING
                break
            node = node[part]
        if node is not _MISSING and node is not None:
            return node
    return _MISSING


def _canonical(value: Any) -> str:
    """Order-insensitive, hashable rendering used for equality comparison."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _variant_of(manifest: dict[str, Any]) -> str | None:
    flags = manifest.get("feature_flags")
    variant = flags.get("variant") if isinstance(flags, dict) else None
    if isinstance(variant, ExperimentVariant):
        return variant.value
    return str(variant) if variant else None


def _label(manifest: dict[str, Any], index: int) -> str:
    source = manifest.get(_SOURCE_KEY)
    if source:
        return str(source)
    variant = _variant_of(manifest)
    return f"{variant}#{index}" if variant else f"run#{index}"


def _as_feature_flags(payload: Any) -> FeatureFlags | None:
    """Rebuild a :class:`FeatureFlags` from a manifest dict, or ``None``."""
    if not isinstance(payload, dict):
        return None
    names = {f.name for f in fields(FeatureFlags)}
    if set(payload) != names:
        return None
    values = dict(payload)
    variant = values["variant"]
    if not isinstance(variant, ExperimentVariant):
        try:
            values["variant"] = ExperimentVariant(variant)
        except ValueError:
            return None
    try:
        return FeatureFlags(**values)
    except TypeError:
        return None


def _diff_flags(left: Any, right: Any) -> tuple[str, ...]:
    """Behaviour-flag names that differ between two manifest flag payloads.

    Uses the authoritative :func:`jobrec.orchestration.feature_flags.flag_diff`
    whenever both payloads round-trip to a :class:`FeatureFlags`; otherwise falls
    back to a total dict comparison over the union of keys (``variant`` excluded,
    exactly as ``flag_diff`` does).
    """
    a = _as_feature_flags(left)
    b = _as_feature_flags(right)
    if a is not None and b is not None:
        return tuple(sorted(flag_diff(a, b)))
    left_map = left if isinstance(left, dict) else {}
    right_map = right if isinstance(right, dict) else {}
    keys = (set(left_map) | set(right_map)) - {"variant"}
    return tuple(sorted(
        key for key in keys
        if left_map.get(key, _MISSING) != right_map.get(key, _MISSING)
    ))


# ---------------------------------------------------------------- field checking
def _check_field(
    name: str,
    paths: tuple[str, ...],
    runs: list[_Run],
    scope: str,
) -> FieldFinding:
    found: dict[str, Any] = {}
    missing: list[str] = []
    groups: dict[str | None, list[tuple[str, Any]]] = {}
    for run in runs:
        value = _lookup(run.manifest, paths)
        key = run.variant if scope == "variant" else None
        entries = groups.setdefault(key, [])
        if value is _MISSING:
            missing.append(run.label)
            entries.append((run.label, _MISSING))
        else:
            found[run.label] = value
            entries.append((run.label, value))

    if not found:
        # Nothing recorded this field anywhere: unverifiable, but not a mismatch.
        return FieldFinding(
            field=name, consistent=True, available=False, values={},
            missing_in=tuple(missing), scope=scope,
        )

    consistent = True
    for entries in groups.values():
        if len(entries) < 2:
            continue
        if any(value is _MISSING for _, value in entries):
            consistent = False
            break
        if len({_canonical(value) for _, value in entries}) > 1:
            consistent = False
            break

    return FieldFinding(
        field=name, consistent=consistent, available=True, values=found,
        missing_in=tuple(missing), scope=scope,
    )


def _check_flags(
    runs: list[_Run],
    target_flag_set: set[str] | frozenset[str] | None,
) -> tuple[FlagFinding, ...]:
    findings: list[FlagFinding] = []
    target = set(target_flag_set) if target_flag_set is not None else None
    for left, right in combinations(runs, 2):
        diff = _diff_flags(left.manifest.get("feature_flags"),
                           right.manifest.get("feature_flags"))
        same_variant = left.variant is not None and left.variant == right.variant
        if same_variant:
            consistent = not diff
            reason = ("identical variant must resolve identical feature flags"
                      if consistent else
                      "repeats of the same variant resolved different feature flags")
        elif target is None:
            consistent = True
            reason = "no target mechanism given; cross-variant flag differences unconstrained"
        elif not diff:
            consistent = False
            reason = ("no feature-flag difference recorded, so the ablation delta is not "
                      "attributable to the target mechanism")
        elif set(diff) <= target:
            consistent = True
            reason = f"only target-mechanism flags differ (target: {', '.join(sorted(target))})"
        else:
            outside = ", ".join(sorted(set(diff) - target))
            consistent = False
            reason = f"flags outside the target mechanism differ: {outside}"
        findings.append(FlagFinding(
            left=left.label, right=right.label, differing_flags=diff,
            consistent=consistent, reason=reason,
        ))
    return tuple(findings)


def _stamp(runs: list[_Run], report: ConsistencyReport) -> None:
    """Write the consistency result flags into each run manifest (R15.3)."""
    for run in runs:
        run.manifest["consistency"] = {
            "consistent": report.consistent,
            "flags": dict(report.flags),
            "mismatched_fields": list(report.mismatched_fields),
            "unavailable_fields": list(report.unavailable_fields),
            "compared_runs": list(report.run_labels),
            "target_flag_set": (
                sorted(report.target_flag_set) if report.target_flag_set is not None else None
            ),
        }


# ------------------------------------------------------------------- public API
def check_consistency(
    manifests: list[dict[str, Any]],
    target_flag_set: set[str] | frozenset[str] | None = None,
    *,
    stamp: bool = True,
) -> ConsistencyReport:
    """Verify that the compared runs share catalog, scenarios, prompts and settings.

    Args:
        manifests: ``run_manifest.json`` payloads of the runs being compared.
        target_flag_set: For an ablation pair, the mechanism's flag group (e.g.
            ``MEMORY_FLAGS``). When given, cross-variant pairs must differ in a
            non-empty subset of it and in nothing else (R32.7). When ``None``,
            cross-variant flag differences are not constrained.
        stamp: When true (the default), write the resulting consistency flags
            into each manifest dict under ``"consistency"`` (R15.3).

    Returns:
        A :class:`ConsistencyReport`. Never raises; use
        :func:`require_consistent` for the gate behaviour.
    """
    runs = [
        _Run(label=_label(manifest, index), variant=_variant_of(manifest), manifest=manifest)
        for index, manifest in enumerate(manifests)
        if isinstance(manifest, dict)
    ]

    field_findings = [
        _check_field(name, paths, runs, "all") for name, paths in _FIELD_PATHS.items()
    ]
    field_findings += [
        _check_field(name, paths, runs, "variant")
        for name, paths in _VARIANT_SCOPED_PATHS.items()
    ]
    flag_findings = _check_flags(runs, target_flag_set)

    flags: dict[str, bool] = {f.field: f.consistent for f in field_findings}
    flags["feature_flags"] = all(f.consistent for f in flag_findings)
    mismatched = tuple(f.field for f in field_findings if not f.consistent)
    unavailable = tuple(f.field for f in field_findings if not f.available)

    report = ConsistencyReport(
        consistent=not mismatched and flags["feature_flags"],
        run_labels=tuple(run.label for run in runs),
        field_findings=tuple(field_findings),
        flag_findings=flag_findings,
        flags=flags,
        mismatched_fields=mismatched,
        unavailable_fields=unavailable,
        target_flag_set=(
            tuple(sorted(target_flag_set)) if target_flag_set is not None else None
        ),
    )
    if stamp:
        _stamp(runs, report)
    return report


def require_consistent(
    manifests: list[dict[str, Any]],
    target_flag_set: set[str] | frozenset[str] | None = None,
) -> ConsistencyReport:
    """Run the gate: raise :class:`ConsistencyError` unless every run matches.

    Always stamps the consistency flags into each manifest first, so the failure
    is recorded on the affected runs as well as raised (R15.2, R15.3).
    """
    report = check_consistency(manifests, target_flag_set)
    if not report.consistent:
        raise ConsistencyError(report)
    return report


def load_run_manifests(experiment_dir: str | Path) -> list[dict[str, Any]]:
    """Load every ``run_manifest.json`` under ``experiment_dir``.

    Each returned dict remembers its source file under a private key so
    :func:`save_run_manifests` can write the stamped flags back.
    """
    manifests: list[dict[str, Any]] = []
    for path in sorted(Path(experiment_dir).rglob("run_manifest.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            payload[_SOURCE_KEY] = str(path)
            manifests.append(payload)
    return manifests


def save_run_manifests(manifests: list[dict[str, Any]]) -> list[Path]:
    """Persist manifests that were loaded from disk, dropping private keys."""
    written: list[Path] = []
    for manifest in manifests:
        source = manifest.get(_SOURCE_KEY) if isinstance(manifest, dict) else None
        if not source:
            continue
        payload = {k: v for k, v in manifest.items() if not k.startswith("_")}
        path = Path(source)
        path.write_text(json.dumps(payload, indent=2, default=str))
        written.append(path)
    return written
