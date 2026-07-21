"""Configuration models and loading.

All thresholds and weights live in configuration, never hard-coded in business
logic. The fully-resolved config is hashed (``config_hash``) so experiment runs
are reproducible and comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .domain.enums import ExperimentVariant, RunMode, UnknownPolicy
from .utils.hashing import stable_hash


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "jobrec-cmjcc"
    environment: str = "local"
    reference_date: str = "2026-01-01"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: ExperimentVariant = ExperimentVariant.FULL
    random_seed: int = 42
    top_k: int = 5
    retrieval_pool_size: int = 50
    repeat_count: int = 3


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    use_prior_dialogue: bool = True
    persist_confirmed_updates: bool = True
    inference_to_long_term: bool = False
    clarification_confidence_threshold: float = 0.72


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explicit_constraint_orchestration: bool = True
    hard_constraint_unknown_default: UnknownPolicy = UnknownPolicy.FAIL
    expired_job_policy: str = "fail"


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "hybrid"
    lexical_weight: float = 0.45
    semantic_weight: float = 0.35
    structured_weight: float = 0.20


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    missing_feature_policy: str = "renormalize"
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "role_match": 0.25,
            "required_skill_match": 0.25,
            "preferred_skill_match": 0.10,
            "location_preference": 0.10,
            "work_mode_preference": 0.10,
            "salary_preference": 0.10,
            "experience_fit": 0.07,
            "industry_preference": 0.03,
        }
    )
    salary_scale: float = 4000.0


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: RunMode = RunMode.DETERMINISTIC
    provider: str = "mock"
    extraction_temperature: float = 0.0
    response_temperature: float = 0.2
    timeout_seconds: int = 30
    max_retries: int = 2
    save_raw_responses: bool = True


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"
    format: str = "json"
    redact_candidate_text: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def config_hash(self) -> str:
        """Stable hash of the fully-resolved configuration."""
        return stable_hash(self.model_dump(mode="json"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, base_dir: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, merged over ``base.yaml``.

    If ``path`` is None, the default :class:`AppConfig` is returned.
    """
    if path is None:
        return AppConfig()

    path = Path(path)
    base_dir = Path(base_dir) if base_dir else path.parent

    data: dict[str, Any] = {}
    base_file = base_dir / "base.yaml"
    if base_file.exists() and base_file.resolve() != path.resolve():
        with base_file.open() as fh:
            data = yaml.safe_load(fh) or {}

    with path.open() as fh:
        override = yaml.safe_load(fh) or {}

    merged = _deep_merge(data, override)
    return AppConfig.model_validate(merged)
