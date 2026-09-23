from __future__ import annotations

import secrets
from functools import lru_cache

from fastapi import Header

from .settings import get_settings
from .storage import LocalStorage, Storage


@lru_cache
def get_storage() -> Storage:
    return LocalStorage(get_settings().storage_root)


def get_actor(x_operator: str | None = Header(default=None, max_length=80)) -> str:
    """Operator name for the audit trail (no authentication in this local prototype; see SRS_AUDIT A-12)."""
    return (x_operator or "operator").strip()[:80] or "operator"


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"
