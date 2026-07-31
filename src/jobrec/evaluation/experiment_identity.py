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
from urllib.parse import urlsplit

from .. import CODE_VERSION
from ..utils.hashing import sha256_of_bytes, stable_hash

#: Packages whose source determines what a run and its analysis produce. ``jobrec`` is
#: the system under evaluation; ``jobrec_eval`` is the pipeline that turns run bundles
#: into the reported numbers. Both are named by the code identity, so both count.
SOURCE_PACKAGES: tuple[str, ...] = ("jobrec", "jobrec_eval")

#: The package whose source can change what a RUN produces.
EXECUTION_PACKAGE = "jobrec"

#: Modules OUTSIDE :data:`EXECUTION_PACKAGE` that the run path nevertheless imports, so
#: their source can change a run bundle. ``jobrec_eval.simulated_user`` is the answerer the
#: clarification loop feeds back into the session, and it pulls in
#: ``jobrec_eval.scenarios``; both therefore decide run outcomes despite living in the
#: analysis package.
#:
#: This allowlist exists because splitting the fingerprint by PACKAGE would be wrong:
#: ``jobrec_eval`` is mostly analysis code that cannot touch a bundle, but not entirely.
#: ``tests/unit/test_experiment_identity_split.py`` statically scans the execution package
#: for ``jobrec_eval`` imports and fails if this list drifts, so it cannot rot.
EXECUTION_EXTRA_MODULES: tuple[str, ...] = (
    "jobrec_eval/simulated_user.py",
    "jobrec_eval/scenarios.py",
)

#: Keys of the dict returned by :func:`code_identity`, in manifest order.
#:
#: ``source_fingerprint`` covers both packages and stays the single "what code was this"
#: value. ``execution_fingerprint`` and ``analysis_fingerprint`` split it by what the code
#: can actually affect, which is what lets an analysis-only edit be re-run over saved
#: bundles without invalidating them: the experiment id is derived from the EXECUTION
#: fingerprint alone (see :func:`experiment_id`). Before the split, editing a report
#: renderer changed the id of an experiment it could not possibly have influenced, which
#: forced an otherwise pointless re-run.
CODE_IDENTITY_FIELDS: tuple[str, ...] = (
    "code_version",
    "commit_hash",
    "git_dirty",
    "source_fingerprint",
    "execution_fingerprint",
    "analysis_fingerprint",
)

#: The file whose presence marks a directory as holding a COMPLETE experiment. The runner
#: writes it after every run has landed, so a crashed run leaves no manifest and may be
#: re-run freely.
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"

#: Keys of the dict returned by :func:`runtime_identity`, in manifest order.
#:
#: These are the run inputs that live OUTSIDE the source tree and outside the resolved
#: config, so neither the code fingerprint nor ``config_hash`` moves when they change:
#:
#: * ``catalog_hash`` -- the jobs that were searched. Editing the catalog changes every
#:   ranking and every relevance grade, but it is a data file, not source.
#: * ``prompt_hash`` -- the prompt texts. They decide what an LLM was asked, and they are
#:   templates rather than config values.
#: * ``llm_mode`` / ``llm_provider`` -- how LLM-dependent behaviour was executed. These do
#:   sit in the config, and are repeated here so the runtime block is readable on its own.
#: * ``llm_model`` / ``llm_endpoint`` -- read from the ENVIRONMENT by
#:   :class:`jobrec.llm.remote_provider.RemoteLLMProvider`, so before this block a hybrid
#:   batch answered by one model and a hybrid batch answered by another shared one
#:   experiment id and the overwrite guard read the second as a re-run of the first.
RUNTIME_IDENTITY_FIELDS: tuple[str, ...] = (
    "catalog_hash",
    "prompt_hash",
    "llm_mode",
    "llm_provider",
    "llm_model",
    "llm_endpoint",
)


def endpoint_identity(endpoint: str | None) -> str | None:
    """A credential-free identity for an endpoint: ``scheme://host[:port]/path``.

    Keeps everything that says WHICH backend answered and drops everything that could
    carry a secret. A base URL is the one run input that can embed a credential -- both
    ``https://key:secret@host/v1`` and ``https://host/v1?api-key=...`` are accepted by
    OpenAI-compatible clients -- so userinfo, query and fragment are removed. What remains
    is parsed structure, not a substring of the input, so a token cannot survive by hiding
    in a part that was merely not looked at.

    The PATH is kept, normalised. An earlier version reduced the endpoint to its host,
    which was safe but too lossy: OpenAI-compatible deployments are routinely distinguished
    by path alone (``/v1`` versus ``/compatible-mode/v1``, or one gateway fronting several
    model deployments), so two genuinely different backends collided on one identity and the
    experiment id could not tell them apart. Scheme and port are kept for the same reason.
    """
    if not endpoint:
        return None
    raw = str(endpoint).strip()
    if not raw:
        return None
    # A bare "host:port" parses as scheme:path, so give it an authority to live in.
    parsed = urlsplit(raw if "//" in raw else f"//{raw}")
    host = parsed.hostname  # drops userinfo and lowercases; never the password
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:  # malformed port -- the rest still identifies the backend
        port = None
    authority = f"{host}:{port}" if port else host
    # Normalised path: collapse repeated separators and drop a trailing one, so
    # ``/v1``, ``/v1/`` and ``//v1`` are one endpoint rather than three.
    segments = [s for s in parsed.path.split("/") if s]
    path = "/" + "/".join(segments) if segments else ""
    scheme = (parsed.scheme or "https").lower()
    return f"{scheme}://{authority}{path}"


def runtime_identity(
    *,
    catalog_hash: str,
    prompt_hash: str,
    llm_mode: str,
    llm_provider: str,
    llm_model: str | None = None,
    llm_endpoint: str | None = None,
) -> dict[str, Any]:
    """The run inputs that are neither source code nor resolved config.

    Returned as a dict so the same value both enters :func:`experiment_id` and is recorded
    in the experiment manifest: the id can then be re-derived from the manifest instead of
    being taken on trust.

    ``llm_endpoint`` is reduced by :func:`endpoint_identity` before it is stored, so a
    credential embedded in a base URL never reaches the manifest or the digest while the
    parts that distinguish two deployments survive. The API KEY itself is not a parameter of
    this function and must never become one: it does not identify an experiment (the same
    key answers every run) and it is the one value that must not be written down.
    """
    return {
        "catalog_hash": catalog_hash,
        "prompt_hash": prompt_hash,
        "llm_mode": llm_mode,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_endpoint": endpoint_identity(llm_endpoint),
    }


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
    return stable_hash(_per_file_digests())


@lru_cache(maxsize=1)
def _per_file_digests() -> dict[str, str]:
    """``relative posix path -> sha256`` for every shipped source file, line-normalised."""
    per_file: dict[str, str] = {}
    for rel, path in _package_source_files():
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        per_file[rel] = sha256_of_bytes(raw.replace(b"\r\n", b"\n"))
    return per_file


def is_execution_source(relative_path: str) -> bool:
    """Whether a source file can change what a RUN produces.

    True for the execution package and for the explicitly allowlisted analysis modules the
    run path imports (:data:`EXECUTION_EXTRA_MODULES`).
    """
    return (relative_path.startswith(f"{EXECUTION_PACKAGE}/")
            or relative_path in EXECUTION_EXTRA_MODULES)


@lru_cache(maxsize=1)
def execution_fingerprint() -> str:
    """Digest over the sources that can change a run bundle.

    This is what :func:`experiment_id` is derived from, so two runs share an id exactly
    when the code that could have influenced their bundles is identical.
    """
    digests = _per_file_digests()
    return stable_hash({rel: digest for rel, digest in digests.items()
                        if is_execution_source(rel)})


@lru_cache(maxsize=1)
def analysis_fingerprint() -> str:
    """Digest over the sources that can only change the ANALYSIS of saved bundles.

    Recorded, never part of the experiment id: an edit here is re-runnable over existing
    bundles, which is the whole point of separating it.
    """
    digests = _per_file_digests()
    return stable_hash({rel: digest for rel, digest in digests.items()
                        if not is_execution_source(rel)})


@lru_cache(maxsize=1)
def _identity_items() -> tuple[tuple[str, Any], ...]:
    """Cached identity pairs: probing git and hashing the tree once per process."""
    return (
        ("code_version", CODE_VERSION),
        ("commit_hash", commit_hash()),
        ("git_dirty", git_dirty()),
        ("source_fingerprint", source_fingerprint()),
        ("execution_fingerprint", execution_fingerprint()),
        ("analysis_fingerprint", analysis_fingerprint()),
    )


def code_identity() -> dict[str, Any]:
    """The identity of the source code running right now (see the module docstring)."""
    return dict(_identity_items())


def reset_code_identity_cache() -> None:
    """Forget the cached identity. For tests, and for any caller that edits sources."""
    for fn in (commit_hash, git_dirty, _per_file_digests, source_fingerprint,
               execution_fingerprint, analysis_fingerprint, _identity_items):
        fn.cache_clear()


def experiment_id(
    *,
    variants: list[str],
    scenario_ids: list[str],
    config_hash: str,
    identity: dict[str, Any] | None = None,
    scenarios_fingerprint: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> str:
    """The content-addressed experiment id: ``exp-<12 hex>``.

    Deterministic in the experiment inputs AND in the code identity, so:

    * re-running unchanged code over unchanged inputs yields the same id (idempotent), and
    * changing the code that could have influenced the bundles yields a different id, which
      is what stops a new official run from landing on top of an older one.

    Derived from the EXECUTION fingerprint, not from the whole tree. Editing a report
    renderer or a metric cannot change what a bundle contains, so it must not change the
    id of the experiment that produced it -- otherwise every analysis fix appears to
    invalidate the run, and re-running an expensive batch looks mandatory when re-running
    the analysis over saved bundles is both sufficient and correct. The analysis identity
    is recorded separately in the manifests.

    ``scenarios_fingerprint`` is a digest of the scenarios' CONTENT, not just their ids.
    Without it, editing a scenario -- its turns, its expectations, or the authoritative
    reference the relevance oracle grades against -- left the id unchanged, so two
    genuinely different experiments collided on one identity and the overwrite guard read
    the second as a re-run of the first. The scenario ids are kept alongside it so a
    changed SET is still distinguishable from changed CONTENT when reading the inputs.

    ``runtime`` is :func:`runtime_identity`: the run inputs that are neither source nor
    resolved config -- the catalog, the prompts, and the LLM backend named by the
    environment. Without it the id was blind to all three, so re-pointing
    ``JOBREC_LLM_MODEL`` at a different model, or editing the job catalog, produced a
    genuinely different experiment under the OLD id, and the overwrite guard classified it
    as an idempotent re-run. It carries no credential: see :func:`endpoint_host`.
    """
    identity = identity or code_identity()
    return "exp-" + stable_hash({
        "variants": list(variants),
        "scenarios": list(scenario_ids),
        "scenarios_fingerprint": scenarios_fingerprint,
        "config": config_hash,
        # Only the fields that describe the code that can affect a RUN.
        "code": {
            "code_version": identity.get("code_version"),
            "execution_fingerprint": identity.get("execution_fingerprint"),
        },
        "runtime": dict(runtime) if runtime else None,
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
