"""The execution/analysis fingerprint split, and the allowlist that keeps it honest.

``experiment_id`` is derived from the EXECUTION fingerprint so that an analysis-only edit
does not invalidate an experiment it could not have influenced. Before the split, editing a
report renderer changed the id of a finished run, which made an expensive re-run look
mandatory when re-running the analysis over the saved bundles was both sufficient and
correct.

The split cannot be done by package. ``jobrec_eval`` is mostly analysis code that cannot
touch a bundle -- but not entirely: the clarification loop imports
``jobrec_eval.simulated_user`` and feeds its answers back into the live session, so that
module decides run outcomes. Splitting on the package name would drop it from the execution
fingerprint and let a change to the answerer silently reuse an older experiment id.

Hence :data:`EXECUTION_EXTRA_MODULES`, and hence this test: it STATICALLY scans the
execution package for imports of the analysis package and fails if the allowlist drifts
from what the run path actually depends on.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from jobrec.evaluation.experiment_identity import (
    CODE_IDENTITY_FIELDS,
    EXECUTION_EXTRA_MODULES,
    EXECUTION_PACKAGE,
    SOURCE_PACKAGES,
    analysis_fingerprint,
    code_identity,
    execution_fingerprint,
    experiment_id,
    is_execution_source,
    source_fingerprint,
)

_SRC = Path(__file__).resolve().parents[2] / "src"
_ANALYSIS_PACKAGE = next(p for p in SOURCE_PACKAGES if p != EXECUTION_PACKAGE)


def _imported_analysis_modules() -> set[str]:
    """``jobrec_eval/<module>.py`` paths that the execution package imports, found by AST.

    Static rather than dynamic: an import inside a rarely-taken branch still changes what a
    run can produce, and would be missed by inspecting ``sys.modules`` after a happy-path
    call.
    """
    found: set[str] = set()
    for path in (_SRC / EXECUTION_PACKAGE).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                if name == _ANALYSIS_PACKAGE or name.startswith(f"{_ANALYSIS_PACKAGE}."):
                    parts = name.split(".")
                    if len(parts) >= 2:
                        found.add(f"{parts[0]}/{parts[1]}.py")
    return found


def _transitive_analysis_modules(seeds: set[str]) -> set[str]:
    """``seeds`` plus every analysis module they import, since those also affect runs."""
    pending = list(seeds)
    seen: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = _SRC / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package = rel.split("/")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                candidate = f"{package}/{node.module.split('.')[0]}.py"
                if (_SRC / candidate).exists():
                    pending.append(candidate)
    return seen


def test_allowlist_matches_what_the_run_path_actually_imports() -> None:
    """The allowlist is neither short (a silent id collision) nor long (a needless re-run).

    **Validates: Requirements 16.1, 11.1**
    """
    required = _transitive_analysis_modules(_imported_analysis_modules())
    assert required, "the scan found no analysis imports at all; the scanner is broken"
    assert set(EXECUTION_EXTRA_MODULES) == required, (
        f"EXECUTION_EXTRA_MODULES has drifted from the real dependency set.\n"
        f"  missing (a run-affecting module would be excluded): "
        f"{sorted(required - set(EXECUTION_EXTRA_MODULES))}\n"
        f"  extra (an analysis-only module would force re-runs): "
        f"{sorted(set(EXECUTION_EXTRA_MODULES) - required)}"
    )


def test_the_simulated_user_counts_as_execution_source() -> None:
    """The answerer the clarification loop feeds back is execution source, not analysis.

    Pinned explicitly because it is the counterexample to a package-based split.

    **Validates: Requirements 16.1**
    """
    assert is_execution_source("jobrec_eval/simulated_user.py")
    assert is_execution_source("jobrec/orchestration/orchestrator.py")
    assert not is_execution_source("jobrec_eval/report.py")
    assert not is_execution_source("jobrec_eval/metrics.py")


def test_the_three_fingerprints_are_distinct_and_recorded() -> None:
    """Execution and analysis digests differ from each other and from the combined one.

    **Validates: Requirements 11.1, 16.1**
    """
    identity = code_identity()
    for field in CODE_IDENTITY_FIELDS:
        assert field in identity, field
    assert identity["execution_fingerprint"] == execution_fingerprint()
    assert identity["analysis_fingerprint"] == analysis_fingerprint()
    assert identity["source_fingerprint"] == source_fingerprint()
    assert len({identity["execution_fingerprint"], identity["analysis_fingerprint"],
                identity["source_fingerprint"]}) == 3


def test_experiment_id_ignores_analysis_only_changes() -> None:
    """An analysis-only edit leaves the experiment id alone; an execution edit moves it.

    This is the property that decides whether an expensive batch has to be repeated.

    **Validates: Requirements 16.1**
    """
    args = {"variants": ["full"], "scenario_ids": ["SC-A-01"], "config_hash": "cfg"}
    base = code_identity()

    analysis_edit = {**base, "analysis_fingerprint": "changed",
                     "source_fingerprint": "changed-too"}
    assert experiment_id(**args, identity=analysis_edit) == experiment_id(
        **args, identity=base)

    execution_edit = {**base, "execution_fingerprint": "changed"}
    assert experiment_id(**args, identity=execution_edit) != experiment_id(
        **args, identity=base)


def test_every_shipped_source_file_is_classified() -> None:
    """No source file is silently in neither digest.

    A file counted in neither would be invisible to both identities, so a change to it
    would leave every recorded fingerprint untouched.

    **Validates: Requirements 11.1**
    """
    files = [f"{package}/{path.relative_to(_SRC / package).as_posix()}"
             for package in SOURCE_PACKAGES
             for path in (_SRC / package).rglob("*.py")]
    assert files
    for rel in files:
        # Total by construction, but assert it so a future third package cannot fall
        # through a predicate that only knows about two.
        assert is_execution_source(rel) or re.match(r"^jobrec_eval/", rel), rel
