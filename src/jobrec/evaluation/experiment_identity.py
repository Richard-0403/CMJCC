"""Code identity of an experiment, the experiment id, and the overwrite guard.

Why this module exists
---------------------
``experiment_id`` used to be a digest of the experiment INPUTS only (the variant list,
the scenario ids and the resolved config hash). The source code that produced the runs
was not part of it, so two official runs from *different* code over the *same* inputs
shared one id: the second run reused the first run's directory and silently replaced it.
That is exactly how a pre-fix baseline was destroyed (observed on
``exp-301060a1899d``), which makes a before/after comparison unperformable and breaks the
"frozen, reproducible thesis artifact" promise.

Two things fix it, and both live here so there is one implementation of each:

1. :func:`experiment_id` mixes the CODE IDENTITY into the digest, so different code over
   identical inputs lands in a different directory by construction, while a re-run of
   unchanged code is idempotent (same inputs + same code -> same id).
2. :func:`guard_output_dir` refuses to write over a directory that already holds a
   COMPLETE experiment (one with an ``experiment_manifest.json``) unless overwriting was
   asked for explicitly. This covers the general clobber case, including a re-run of
   unchanged code and a hand-picked ``--out-root``.

What "code identity" means here
-------------------------------
* ``code_version`` -- the declared package version (:data:`jobrec.CODE_VERSION`).
* ``commit_hash`` -- ``git rev-parse HEAD``, or ``None`` outside a git checkout.
* ``git_dirty`` -- whether the working tree has uncommitted or untracked changes
  (``None`` when git is unavailable). A dirty tree means ``commit_hash`` does NOT
  identify the code, which is why it is recorded and why the id does not rely on the
  commit hash.
* ``source_fingerprint`` -- a content digest over every ``*.py`` file of the shipped
  packages (:data:`SOURCE_PACKAGES`). This is what actually distinguishes two code
  states, committed or not, so it is what the id is built from.

Only ``code_version`` and ``source_fingerprint`` enter :func:`experiment_id`: the commit
hash would move the id when a commit merely records source that is already on disk, and
``git_dirty`` is already implied by the fingerprint. Both are still recorded in the
manifests so two artifacts can be told apart offline.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import CODE_VERSION
from ..utils.hashing import sha256_of_bytes, stable_hash

#: Packages whose source determines what a run and its analysis produce. ``jobrec`` is
#: the system under evaluation; ``jobrec_eval`` is the pipeline that turns run bundles
#: into the reported numbers. Both are named by the experiment id, so both count.
SOURCE_PACKAGES: tuple[str, ...] = ("jobrec", "jobrec_eval")

#: Keys of the dict returned by :func:`code_identity`, in manifest order.
CODE_IDENTITY_FIELDS: tuple[str, ...] = (
    "code_version",
    "commit_hash",
    "git_dirty",
    "source_fingerprint",
)

#: The file whose presence marks a directory as holding a COMPLETE experiment. The runner
#: writes it after every run has landed, so a crashed run leaves no manifest and may be
#: re-run freely.
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"


def _git(*args: str) -> str | None:
    """Run a git command in the repo and return its stdout, or ``None`` on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=str(Path(__file__).resolve().parent),
        )
    except Exception:  # noqa: BLE001 - git missing / not a repo / timeout
        return None
    if out.returncode != 0:
        return None
    return out.stdout


@lru_cache(maxsize=1)
def commit_hash() -> str | None:
    """Current git commit hash, or ``None`` when git/the checkout is unavailable."""
    out = _git("rev-parse", "HEAD")
    return (out.strip() or None) if out is not None else None


@lru_cache(maxsize=1)
def git_dirty() -> bool | None:
    """Whether the working tree differs from ``HEAD`` (untracked files included).

    ``None`` when git is unavailable, i.e. "unknown" rather than a misleading ``False``.
    """
    out = _git("status", "--porcelain")
    return bool(out.strip()) if out is not None else None


#: Directory that CONTAINS the source packages (``src/`` in a checkout, ``site-packages``
#: in an installed environment). Derived from this file's own location so the fingerprint
#: never has to import a package -- importing ``jobrec_eval`` would drag pandas in, and a
#: package that happened to be unimportable would silently change the fingerprint.
_PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def _package_source_files() -> list[tuple[str, Path]]:
    """``(relative posix path, absolute path)`` of every ``*.py`` in :data:`SOURCE_PACKAGES`.

    Sorted by the relative path so the traversal order cannot affect the digest. A package
    directory that is not present is skipped rather than raising: the fingerprint is
    provenance, never a reason for an experiment to fail.
    """
    files: list[tuple[str, Path]] = []
    for name in SOURCE_PACKAGES:
        root = _PACKAGE_PARENT / name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            files.append((f"{name}/{path.relative_to(root).as_posix()}", path))
    files.sort(key=lambda item: item[0])
    return files


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Content digest over the shipped Python sources of :data:`SOURCE_PACKAGES`.

    Line endings are normalised to ``\\n`` before hashing so the same commit checked out
    on Windows and on Linux fingerprints identically; anything else that differs in the
    bytes of a source file changes the fingerprint, committed or not.
    """
    per_file: dict[str, str] = {}
    for rel, path in _package_source_files():
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        per_file[rel] = sha256_of_bytes(raw.replace(b"\r\n", b"\n"))
    return stable_hash(per_file)


@lru_cache(maxsize=1)
def _identity_items() -> tuple[tuple[str, Any], ...]:
    """Cached identity pairs: probing git and hashing the tree once per process."""
    return (
        ("code_version", CODE_VERSION),
        ("commit_hash", commit_hash()),
        ("git_dirty", git_dirty()),
        ("source_fingerprint", source_fingerprint()),
    )


def code_identity() -> dict[str, Any]:
    """The identity of the source code running right now (see the module docstring)."""
    return dict(_identity_items())


def reset_code_identity_cache() -> None:
    """Forget the cached identity. For tests, and for any caller that edits sources."""
    for fn in (commit_hash, git_dirty, source_fingerprint, _identity_items):
        fn.cache_clear()


def experiment_id(
    *,
    variants: list[str],
    scenario_ids: list[str],
    config_hash: str,
    identity: dict[str, Any] | None = None,
) -> str:
    """The content-addressed experiment id: ``exp-<12 hex>``.

    Deterministic in the experiment inputs AND in the code identity, so:

    * re-running unchanged code over unchanged inputs yields the same id (idempotent), and
    * changing the code yields a different id, which is what stops a new official run from
      landing on top of an older one.
    """
    identity = identity or code_identity()
    return "exp-" + stable_hash({
        "variants": list(variants),
        "scenarios": list(scenario_ids),
        "config": config_hash,
        # Only the two fields that describe the code CONTENT -- see the module docstring.
        "code": {
            "code_version": identity.get("code_version"),
            "source_fingerprint": identity.get("source_fingerprint"),
        },
    })[:12]


class ExperimentOverwriteError(RuntimeError):
    """Raised instead of silently replacing an existing, complete experiment artifact."""


def _recorded_identity(manifest_path: Path) -> dict[str, Any] | None:
    """The code identity recorded in an existing manifest, or ``None`` if unreadable."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    return {key: manifest.get(key) for key in CODE_IDENTITY_FIELDS}


def _describe(identity: dict[str, Any] | None) -> str:
    if identity is None:
        return "unrecorded (manifest missing or unreadable)"
    fingerprint = identity.get("source_fingerprint")
    short = str(fingerprint)[:12] if fingerprint else "unknown"
    return (f"code_version={identity.get('code_version')!r} "
            f"commit_hash={identity.get('commit_hash')!r} "
            f"git_dirty={identity.get('git_dirty')!r} "
            f"source_fingerprint={short}")


def guard_output_dir(
    target: Path,
    *,
    manifest_name: str = EXPERIMENT_MANIFEST_FILENAME,
    identity: dict[str, Any] | None = None,
    allow_overwrite: bool = False,
    overwrite_flag: str = "--allow-overwrite",
) -> None:
    """Refuse to write into ``target`` when it already holds a complete experiment.

    ``target`` is the directory about to be written (a run-bundle experiment directory or
    an analysis output directory) and ``manifest_name`` is the manifest inside it,
    relative to ``target``, whose presence means "complete". An absent manifest (a fresh
    or crashed directory) is never an obstacle.

    Args:
        allow_overwrite: When true the existing artifact is reused/overwritten on purpose
            -- this is how an intentional idempotent re-run is expressed.
        overwrite_flag: The flag name quoted in the error message, so the message is
            actionable for whichever entry point raised it.

    Raises:
        ExperimentOverwriteError: ``target`` holds a complete experiment and
            ``allow_overwrite`` is false. The message says whether the recorded code
            identity differs from the current one, i.e. whether overwriting would destroy
            a baseline produced by DIFFERENT code.
    """
    manifest_path = target / manifest_name
    if not manifest_path.is_file():
        return
    if allow_overwrite:
        return

    identity = identity or code_identity()
    recorded = _recorded_identity(manifest_path)
    same_code = recorded is not None and all(
        recorded.get(key) == identity.get(key) for key in ("code_version", "source_fingerprint")
    )
    if same_code:
        verdict = ("The recorded code identity MATCHES the code running now, so this would "
                   "be a re-run of the same experiment.")
    else:
        verdict = ("The recorded code identity DIFFERS from the code running now: "
                   "overwriting would destroy a baseline produced by other code and make "
                   "the before/after comparison unperformable.")
    raise ExperimentOverwriteError(
        f"refusing to overwrite the complete experiment in {target}: {verdict}\n"
        f"  recorded: {_describe(recorded)}\n"
        f"  current:  {_describe(identity)}\n"
        f"Choose one:\n"
        f"  - write somewhere else (a different --out-root/--out-dir), keeping {target} intact;\n"
        f"  - move or rename {target} first if you want it archived;\n"
        f"  - pass {overwrite_flag} to overwrite it on purpose."
    )
