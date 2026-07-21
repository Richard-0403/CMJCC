"""API dependencies: a process-wide AppService singleton."""

from __future__ import annotations

import os
from functools import lru_cache

from ..app_service import AppService, build_default_service
from ..config import load_config


@lru_cache(maxsize=1)
def get_service() -> AppService:
    config_path = os.environ.get("JOBREC_CONFIG", "configs/experiment_full.yaml")
    catalog_path = os.environ.get("JOBREC_CATALOG", "data/processed/jobs.jsonl")
    config = load_config(config_path, base_dir=os.path.dirname(config_path) or "configs")
    return build_default_service(config, catalog_path=catalog_path)
