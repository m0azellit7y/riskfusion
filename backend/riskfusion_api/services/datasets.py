"""Register a simulated dataset version into PostgreSQL: dataset row, session rows, DR-4 labels, splits."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models import DatasetVersion
from .audit import audit

_SESSION_COLS = (
    "id, source, status, dataset_version, behavior_profile, lighting, webcam_class, room_noise, "
    "connection_stability, eyewear, head_covering, baseline_movement, duration_s, channels_outage, "
    "is_rehearsal, created_at, updated_at"
)


def register_simulated(db: Session, data_root: Path, dataset_version: str) -> dict[str, Any]:
    root = data_root / "synthetic" / dataset_version
    manifest = json.loads((root / "manifest.json").read_text())
    sessions = pd.read_parquet(root / "sessions.parquet")
    labels = pd.read_parquet(root / "labels.parquet")

    dv = db.get(DatasetVersion, dataset_version) or DatasetVersion(id=dataset_version)
    dv.kind = "simulated"
    dv.config_version = manifest["config_version"]
    dv.config_hash = manifest["config_hash"]
    dv.content_hash = manifest["content_hash"]
    dv.seed = manifest["seed"]
    dv.n_sessions = manifest["n_sessions"]
    dv.n_events = manifest["n_events"]
    dv.dead_letter_events = manifest["dead_letter_events"]
    dv.positive_rate = manifest["positive_rate"]
    dv.noise_source = manifest["noise_source"]
    dv.generation_seconds = manifest["generation_seconds"]
    dv.uri = str(root)
    dv.manifest = manifest
    db.add(dv)
    db.flush()

    now = datetime.now(timezone.utc)
    rows = [
        {
            "id": r.session_id,
            "source": "SIMULATED",
            "status": "GENERATED",
            "dataset_version": dataset_version,
            "behavior_profile": r.behavior_profile,
            "lighting": r.lighting,
            "webcam_class": r.webcam_class,
            "room_noise": r.room_noise,
            "connection_stability": r.connection_stability,
            "eyewear": bool(r.eyewear),
            "head_covering": bool(r.head_covering),
            "baseline_movement": float(r.baseline_movement),
            "duration_s": float(r.duration_s),
            "channels_outage": json.dumps(list(r.channels_outage)),
            "is_rehearsal": False,
            "created_at": now,
            "updated_at": now,
        }
        for r in sessions.itertuples(index=False)
    ]
    db.execute(
        text(
            f"INSERT INTO sessions ({_SESSION_COLS}) VALUES (:id, :source, :status, :dataset_version, "
            ":behavior_profile, :lighting, :webcam_class, :room_noise, :connection_stability, :eyewear, "
            ":head_covering, :baseline_movement, :duration_s, CAST(:channels_outage AS JSONB), :is_rehearsal, "
            ":created_at, :updated_at) ON CONFLICT (id) DO NOTHING"
        ),
        rows,
    )
    lrows = [
        {
            "session_id": r.session_id,
            "source": r.source,
            "violation": bool(r.violation),
            "violation_types": list(r.violation_types),
            "intervals": r.intervals,
            "labeler": r.labeler,
            "labeled_at": now,
            "confidence": r.confidence,
        }
        for r in labels.itertuples(index=False)
    ]
    db.execute(
        text(
            "INSERT INTO labels (session_id, source, violation, violation_types, intervals, labeler, labeled_at, "
            "confidence) VALUES (:session_id, :source, :violation, :violation_types, CAST(:intervals AS JSONB), "
            ":labeler, :labeled_at, :confidence) ON CONFLICT (session_id) DO NOTHING"
        ),
        lrows,
    )
    n_splits = 0
    split_manifest = data_root / "splits" / dataset_version / "manifest.json"
    if split_manifest.exists():
        sm = json.loads(split_manifest.read_text())
        splits = pd.read_csv(split_manifest.parent / "splits.csv")
        srows = [
            {"session_id": r.session_id, "split_version": sm["split_version"], "split": r.split}
            for r in splits.itertuples(index=False)
        ]
        db.execute(
            text(
                "INSERT INTO data_splits (session_id, split_version, split) VALUES (:session_id, :split_version, "
                ":split) ON CONFLICT DO NOTHING"
            ),
            srows,
        )
        n_splits = len(srows)
    audit(db, "dataset.register", "dataset", dataset_version, "cli", {"sessions": len(rows), "splits": n_splits})
    db.commit()
    return {"dataset_version": dataset_version, "sessions": len(rows), "labels": len(lrows), "splits": n_splits}
