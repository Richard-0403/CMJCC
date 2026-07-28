"""Run-manifest builder (landing-plan R11).

Captures the reproducibility-relevant environment and provenance for a single
experiment run: source commit (plus a dirty-tree flag and a content fingerprint of
the sources, since a commit hash alone does not identify a dirty working tree),
interpreter and dependency versions, host
hardware summary, the run's content hashes, resolved feature flags, a
non-sensitive API summary, and the DB/migration versions.

The single public entry point, :func:`build_run_manifest`, is pure and total:
it returns a plain JSON-serializable ``dict`` and never raises. Every probe of
the environment (git, package metadata, hardware) is guarded so a missing tool
or package yields ``None``/``"unknown"`` rather than an exception. Secrets and
API keys are NEVER included -- only a redacted API summary derived from the
non-sensitive model manifest and configuration.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from .experiment_identity import commit_hash, git_dirty, source_fingerprint

# Key runtime/eval packages whose versions materially affect reproducibility.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "pydantic",
    "pydantic-settings",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "psycopg",
    "pyyaml",
    "numpy",
    "scikit-learn",
    "httpx",
    "structlog",
    "typer",
    "pandas",
    "scipy",
    "matplotlib",
    "hypothesis",
    "pytest",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a pydantic model / object attr or a mapping, safely."""
    if obj is None:
        return default
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:  # noqa: BLE001 - never let attribute access break the manifest
        return default


def _commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` when git is unavailable.

    Delegates to the single git probe in
    :mod:`jobrec.evaluation.experiment_identity` (cached per process), so the run
    manifest, the experiment manifest and the experiment id can never disagree about
    which commit produced a run.
    """
    return commit_hash()


def _dependency_versions() -> dict[str, str | None]:
    """Best-effort map of tracked package -> installed version (None if absent)."""
    try:
        from importlib import metadata
    except Exception:  # noqa: BLE001 - importlib.metadata should always exist on 3.11
        return {}
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - package not installed / metadata error
            versions[name] = None
    return versions


def _total_memory_bytes() -> int | None:
    """Best-effort total physical memory in bytes; ``None`` when undeterminable."""
    # POSIX sysconf is available without extra dependencies on Linux/macOS. It does not
    # exist at all on Windows, so look it up dynamically rather than calling os.sysconf
    # directly (which would be an AttributeError at runtime and an error under a type
    # checker running with --platform win32).
    sysconf = getattr(os, "sysconf", None)
    if sysconf is not None:
        try:
            page_size = sysconf("SC_PAGE_SIZE")
            phys_pages = sysconf("SC_PHYS_PAGES")
            if page_size > 0 and phys_pages > 0:
                return int(page_size) * int(phys_pages)
        except (ValueError, OSError):
            pass
    # Optional psutil fallback if present in the environment.
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - psutil not installed / probe failed
        return None


def _python_summary() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _host_summary() -> dict[str, Any]:
    try:
        cpu_count = os.cpu_count()
    except Exception:  # noqa: BLE001
        cpu_count = None
    return {
        "system": platform.system() or None,
        "release": platform.release() or None,
        "version": platform.version() or None,
        "machine": platform.machine() or None,
        "processor": platform.processor() or None,
        "cpu_count": cpu_count,
        "total_memory_bytes": _total_memory_bytes(),
    }


def _api_summary(config: Any, model_manifest: dict[str, Any]) -> dict[str, Any]:
    """Non-sensitive API summary. NEVER includes key material.

    Draws the provider/model/mode from the run's model manifest when available
    and falls back to the resolved LLM configuration. Any base URL is reduced to
    its host component so no path, query, or credential is retained.
    """
    llm = _get(config, "llm")
    manifest = model_manifest or {}

    provider = manifest.get("provider") or _get(llm, "provider")
    model = manifest.get("model")
    mode = manifest.get("mode")
    if mode is None:
        raw_mode = _get(llm, "mode")
        # RunMode enum -> its value; plain strings pass through.
        mode = getattr(raw_mode, "value", raw_mode)

    base_url_host: str | None = None
    base_url = manifest.get("base_url")
    if isinstance(base_url, str) and base_url:
        try:
            parsed = urlparse(base_url)
            base_url_host = parsed.hostname or None
        except Exception:  # noqa: BLE001
            base_url_host = None

    return {
        "provider": provider,
        "model": model,
        "mode": mode,
        "base_url_host": base_url_host,
    }


def build_run_manifest(
    config: Any,
    run_record: Any,
    versions: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for a single run.

    Args:
        config: The resolved :class:`~jobrec.config.AppConfig` (or any object /
            mapping exposing an ``llm`` section).
        run_record: The :class:`~jobrec.domain.run_record.RunRecord` carrying the
            run's content hashes, resolved ``feature_flags``, ``model_manifest``,
            ``code_version`` and db/migration versions.
        versions: The mapping returned by ``SqlRepository.versions()`` with keys
            ``db_version`` and ``migration_version``; may be ``None``.

    Returns:
        A plain JSON-serializable ``dict``. This function never raises.
    """
    versions = versions or {}
    model_manifest = _get(run_record, "model_manifest", {}) or {}

    # DB / migration versions: prefer the live ``versions`` probe, fall back to
    # whatever was recorded on the run record.
    db_version = versions.get("db_version")
    if db_version is None:
        db_version = _get(run_record, "db_version")
    migration_version = versions.get("migration_version")
    if migration_version is None:
        migration_version = _get(run_record, "migration_version")

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "commit_hash": _commit_hash(),
        "code_version": _get(run_record, "code_version"),
        # A commit hash alone does not identify the code when the tree is dirty, so record
        # whether it was, plus a content digest of the sources actually on disk. Same
        # fields (and same values) as the experiment manifest's code identity.
        "git_dirty": git_dirty(),
        "source_fingerprint": source_fingerprint(),
        "python": _python_summary(),
        "host": _host_summary(),
        "dependencies": _dependency_versions(),
        "hashes": {
            "config_hash": _get(run_record, "config_hash"),
            "catalog_hash": _get(run_record, "catalog_hash"),
            "prompt_hash": _get(run_record, "prompt_hash"),
        },
        "feature_flags": _get(run_record, "feature_flags", {}) or {},
        "api_summary": _api_summary(config, model_manifest),
        "versions": {
            "db_version": db_version,
            "migration_version": migration_version,
        },
    }
    return manifest
