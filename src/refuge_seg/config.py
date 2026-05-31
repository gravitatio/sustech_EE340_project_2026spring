from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in config or not isinstance(config[name], dict):
        raise ValueError(f"Missing config section: {name}")
    return config[name]

