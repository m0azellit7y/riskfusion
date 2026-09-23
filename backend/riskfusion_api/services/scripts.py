"""Mock-session scripts (configs/mock_scripts.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "mock_scripts.yaml"


@lru_cache
def load_scripts() -> dict[str, Any]:
    return dict(yaml.safe_load(SCRIPTS_PATH.read_text()))


def get_script(script_id: str) -> dict[str, Any]:
    scripts = load_scripts()["scripts"]
    if script_id not in scripts:
        raise KeyError(script_id)
    return {"id": script_id, **scripts[script_id]}
