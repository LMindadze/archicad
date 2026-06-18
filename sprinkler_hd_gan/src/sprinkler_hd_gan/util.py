from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def merge_config(base: dict[str, Any], override: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not override:
        return base
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out
