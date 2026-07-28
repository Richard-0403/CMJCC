"""Unified artifact checksums (R16).

One manifest, one command, every artifact. :func:`write_checksums` walks an
experiment (or evaluation output) directory and records a sha256 digest for
**every** artifact it finds -- the snapshotted inputs (``resolved_config.yaml``,
``catalog.jsonl``, ``scenarios.jsonl``) as well as all outputs (per-run bundles,
manifests, index/failure tables, normalized data, metrics, statistics, plots and
the report) -- into a single ``checksums.json`` at the directory root (R16.1).

This supersedes the two earlier, partial mechanisms it replaces: the runner's
``checksums.sha256`` (which hashed only ``*.json``) and the evaluation
pipeline's ``audit/checksums.sha256`` (``*.csv`` + ``*.json`` only). Both are now
routed through this module so there is exactly one checksum implementation and
one manifest format.

The manifest is a flat, deterministic ``{relative_path: sha256}`` mapping:

* paths are POSIX-relative to the directory root, so the same tree produces the
  same manifest on Windows and Linux,
* keys are sorted, and no timestamp or host detail is recorded, so writing the
  manifest twice over an unchanged tree yields byte-identical output.

When a later stage legitimately rewrites an artifact the manifest already
covers, :func:`restamp_checksums` updates just those entries, so the manifest
describes the final on-disk state instead of an intermediate one. It is the only
sanctioned way to amend a manifest in place; every other artifact keeps the
digest it was first recorded with, so tampering elsewhere is still caught.

:func:`verify_checksums` recomputes the tree and reports every disagreement as a
:class:`ChecksumMismatch` -- ``modified`` (digest changed), ``missing`` (recorded
but absent) or ``untracked`` (present but not recorded) -- naming the offending
artifact so the ``verify`` CLI command can print it and exit non-zero
(R16.2/16.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..utils.hashing import sha256_of_bytes

__all__ = [
    "CHECKSUMS_FILENAME",
    "LEGACY_CHECKSUM_FILENAMES",
    "ChecksumMismatch",
    "MissingChecksumsError",
    "compute_checksums",
    "iter_artifacts",
    "read_checksums",
    "restamp_checksums",
    "sha256_of_file",
    "verify_checksums",
    "write_checksums",
]

#: Name of the unified manifest, written at the root of the artifact directory.
CHECKSUMS_FILENAME = "checksums.json"

#: Superseded partial manifests. They are never hashed (their content is derived
#: from the very tree being hashed) and are removed when a unified manifest is
#: written, so a directory never carries two competing checksum files.
LEGACY_CHECKSUM_FILENAMES: tuple[str, ...] = ("checksums.sha256",)

#: Directories that hold no experiment artifacts (tooling/OS noise only).
_SKIPPED_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
})

#: Files that are OS noise rather than artifacts.
_SKIPPED_FILES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


class MissingChecksumsError(FileNotFoundError):
    """Raised when a directory has no ``checksums.json`` to verify against."""


@dataclass(frozen=True)
class ChecksumMismatch:
    """One disagreement between the recorded manifest and the artifacts on disk.

    ``kind`` is one of ``modified`` (recorded and present, digests differ),
    ``missing`` (recorded but no longer on disk) or ``untracked`` (on disk but
    absent from the manifest).
    """

    artifact: str
    kind: str
    expected: str | None = None
    actual: str | None = None

    def describe(self) -> str:
        """Human-readable one-liner naming the offending artifact (R16.3)."""
        if self.kind == "modified":
            return (
                f"{self.artifact}: content changed "
                f"(expected {self.expected}, found {self.actual})"
            )
        if self.kind == "missing":
            return f"{self.artifact}: recorded in {CHECKSUMS_FILENAME} but missing on disk"
        if self.kind == "untracked":
            return f"{self.artifact}: present on disk but not recorded in {CHECKSUMS_FILENAME}"
        return f"{self.artifact}: {self.kind}"


def sha256_of_file(path: str | Path) -> str:
    """Return the hex sha256 digest of a file's raw bytes.

    Hashes bytes (not decoded text) through the shared
    :func:`jobrec.utils.hashing.sha256_of_bytes` helper, so binary artifacts
    (plots) and text artifacts are treated identically and line-ending
    differences are never silently normalised away.
    """
    return sha256_of_bytes(Path(path).read_bytes())


def iter_artifacts(exp_dir: str | Path) -> Iterator[Path]:
    """Yield every artifact file under ``exp_dir``, sorted and root-relative.

    Walks the whole tree (inputs and outputs alike) and skips only the checksum
    manifests themselves plus tooling/OS noise, so the covered set is "everything
    the experiment wrote" rather than a hand-maintained list of names.
    """
    root = Path(exp_dir)
    excluded_names = {CHECKSUMS_FILENAME, *LEGACY_CHECKSUM_FILENAMES}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _SKIPPED_DIRS.intersection(relative.parts[:-1]):
            continue
        if relative.name in _SKIPPED_FILES or relative.name in excluded_names:
            continue
        paths.append(relative)
    yield from sorted(paths, key=lambda p: p.as_posix())


def compute_checksums(exp_dir: str | Path) -> dict[str, str]:
    """Return ``{posix_relative_path: sha256}`` for every artifact under ``exp_dir``."""
    root = Path(exp_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not an artifact directory: {root}")
    return {
        relative.as_posix(): sha256_of_file(root / relative)
        for relative in iter_artifacts(root)
    }


def write_checksums(exp_dir: str | Path) -> Path:
    """Write the unified ``checksums.json`` for ``exp_dir`` and return its path (R16.1).

    Covers all input and output artifacts in the directory. Any superseded
    ``checksums.sha256`` is removed so a single manifest remains authoritative.
    """
    root = Path(exp_dir)
    checksums = compute_checksums(root)
    target = root / CHECKSUMS_FILENAME
    target.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    for legacy_name in LEGACY_CHECKSUM_FILENAMES:
        legacy = root / legacy_name
        if legacy.exists():
            legacy.unlink()
    return target


def _enclosing_manifest_root(path: Path) -> Path | None:
    """The nearest ancestor directory of ``path`` that carries a manifest, if any."""
    for parent in path.parents:
        if (parent / CHECKSUMS_FILENAME).is_file():
            return parent
    return None


def restamp_checksums(paths: Iterable[str | Path]) -> list[Path]:
    """Re-hash ``paths`` in the manifest that already covers them (R16.1).

    An artifact that is legitimately rewritten AFTER its directory's
    ``checksums.json`` was written leaves a stale digest behind, and
    :func:`verify_checksums` then reports a ``modified`` finding for a tree
    nobody tampered with. This updates exactly the named entries in the nearest
    enclosing manifest so the manifest keeps describing the final on-disk state,
    without re-walking (and thereby silently re-blessing) the rest of the tree:
    a genuinely tampered sibling artifact is still caught.

    Paths with no enclosing manifest, and paths that no longer exist, are
    skipped, so callers can pass whatever they wrote without probing first.
    Returns the manifest files that were rewritten (empty when nothing drifted).
    """
    pending: dict[Path, dict[str, str]] = {}
    for candidate in paths:
        path = Path(candidate).resolve()
        if not path.is_file():
            continue
        root = _enclosing_manifest_root(path)
        if root is None:
            continue
        pending.setdefault(root, {})[path.relative_to(root).as_posix()] = sha256_of_file(path)

    written: list[Path] = []
    for root, updates in pending.items():
        recorded = read_checksums(root)
        if all(recorded.get(artifact) == digest for artifact, digest in updates.items()):
            continue
        recorded.update(updates)
        target = root / CHECKSUMS_FILENAME
        target.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        written.append(target)
    return written


def read_checksums(exp_dir: str | Path) -> dict[str, str]:
    """Load the recorded manifest for ``exp_dir``.

    Raises :class:`MissingChecksumsError` when no manifest exists and
    ``ValueError`` when the manifest is not a ``{path: digest}`` mapping.
    """
    path = Path(exp_dir) / CHECKSUMS_FILENAME
    if not path.is_file():
        raise MissingChecksumsError(f"no {CHECKSUMS_FILENAME} in {exp_dir}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in payload.items()
    ):
        raise ValueError(f"malformed {CHECKSUMS_FILENAME} in {exp_dir}")
    return payload


def verify_checksums(
    exp_dir: str | Path, *, report_untracked: bool = True
) -> list[ChecksumMismatch]:
    """Recompute ``exp_dir`` and return every mismatch, empty when intact (R16.2).

    Findings are ordered by artifact path so output is stable. Set
    ``report_untracked=False`` to ignore files added after the manifest was
    written (they are otherwise reported as ``untracked``).
    """
    root = Path(exp_dir)
    recorded = read_checksums(root)
    actual = compute_checksums(root)

    findings: list[ChecksumMismatch] = []
    for artifact, expected in recorded.items():
        found = actual.get(artifact)
        if found is None:
            findings.append(ChecksumMismatch(artifact, "missing", expected, None))
        elif found != expected:
            findings.append(ChecksumMismatch(artifact, "modified", expected, found))
    if report_untracked:
        for artifact, found in actual.items():
            if artifact not in recorded:
                findings.append(ChecksumMismatch(artifact, "untracked", None, found))
    return sorted(findings, key=lambda finding: (finding.artifact, finding.kind))
