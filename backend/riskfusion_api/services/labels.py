"""Scripted ground truth for mock sessions (DR-2: labels come from the script, never from footage)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models import ExamSession, Label, ScriptEpisode
from .scripts import get_script


def write_mock_label(db: Session, sess: ExamSession, duration_s: float) -> Label:
    script = get_script(sess.script_id or "")
    episodes = db.query(ScriptEpisode).filter_by(session_id=sess.id).order_by(ScriptEpisode.episode_no).all()
    dur_ms = int(duration_s * 1000)
    intervals: list[dict[str, int | str]] = []
    all_shown = True
    for ep in episodes:
        if ep.scheduled_start_ms >= dur_ms:
            all_shown = False  # recording ended before this cue was due
            continue
        if ep.shown_at_ms is None:
            all_shown = False
        end = dur_ms if ep.scheduled_end_ms < 0 else min(ep.scheduled_end_ms, dur_ms)
        intervals.append({"start_ms": int(ep.scheduled_start_ms), "end_ms": int(end), "type": ep.violation_type})
    if not script["violation"]:
        intervals = []  # practice cues in rehearsal/clean scripts are not violations
    violation = bool(script["violation"]) and len(intervals) > 0
    types = sorted({str(i["type"]) for i in intervals})
    label = db.get(Label, sess.id) or Label(session_id=sess.id)
    label.source = "mock"
    label.violation = violation
    label.violation_types = types
    label.intervals = intervals
    label.labeler = f"script:{script['id']}@{sess.script_version}"
    label.labeled_at = datetime.now(timezone.utc)
    # 'certain' only if every scheduled cue was actually shown to the participant
    label.confidence = "certain" if all_shown else "uncertain"
    db.add(label)
    return label
