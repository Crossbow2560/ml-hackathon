"""Load configs/default.yaml (single source of truth for radius, K, thresholds, seeds)."""
from __future__ import annotations

from functools import lru_cache

import yaml

from pathsnap.io import REPO_ROOT


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)
