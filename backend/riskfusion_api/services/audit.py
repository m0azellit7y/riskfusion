from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..models import AuditLog


def audit(
    db: Session,
    action: str,
    entity_type: str,
    entity_id: str | None,
    actor: str = "operator",
    details: dict[str, Any] | None = None,
) -> None:
    db.add(AuditLog(actor=actor, action=action, entity_type=entity_type, entity_id=entity_id, details=details))
