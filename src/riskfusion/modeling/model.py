"""The deployable risk model bundle and the risk_assessment.v1 it produces.

A bundle directory (models/trained/<version>/) contains:
  model.txt            LightGBM booster
  bundle.json          features, calibration map, confidence-band map, operating point, tiers, metadata
Everything needed to score is in these two files, so the API, batch CLI and reports share one code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.contracts import CHANNELS, RISK_SCHEMA, risk_assessment_errors
from riskfusion.features.engine import FEATURE_VERSION, SIGNALS

INTERACTION_CHANNELS = {
    "x_gaze_off_and_foreign": ("attention", "audio_voice"),
    "x_phone_and_pitch_down": ("environment", "attention"),
    "x_multi_face_and_voice": ("presence", "environment", "audio_voice"),
    "x_mismatch_and_face": ("identity", "presence"),
    "x_gaze_off_no_typing": ("attention", "behavioral"),
    "x_paper_and_pitch_down": ("environment", "attention"),
    "x_paste_after_tab_return": ("screen",),
    "x_foreign_and_keys": ("audio_voice", "behavioral"),
}


def feature_channels(name: str) -> tuple[str, ...]:
    """Which channel(s) a feature is computed from (used for missing-channel handling and ablations)."""
    if name == "baseline_available":
        return ("attention",)
    if name.startswith("unk_"):
        return (name[4:],)
    if name in INTERACTION_CHANNELS:
        return INTERACTION_CHANNELS[name]
    base = name[3:] if name.startswith("bn_") else name
    sig = base.split("__")[0]
    if sig in SIGNALS:
        return (SIGNALS[sig][0],)
    if name.startswith("q_luma"):
        return ("presence",)
    return ()


def is_model_feature(name: str) -> bool:
    """Quality (luma) features are excluded: they are not evidence and would break the FR-30 guarantee."""
    return not name.startswith("q_")


def monotone_direction(name: str) -> int:
    """+1 for evidence features, -1 for missing-data shares, 0 for context (duration)."""
    if name.startswith("unk_"):
        return -1
    if name == "duration_min":
        return 0
    return 1


SAFE_HIGH = 1e6  # imputation for decreasing features: at least as large as any real value (all are bounded)


def safe_value(direction: int) -> float:
    """The value that can only LOWER risk for a monotone feature: 0 for increasing (all evidence features are
    >= 0), a very large value for decreasing ones."""
    return SAFE_HIGH if direction < 0 else 0.0


def prepare(X: pd.DataFrame, directions: dict[str, int] | None = None) -> pd.DataFrame:
    """Missing values are replaced by their risk-lowering value, so NaN never reaches the trees."""
    if directions is None:
        return X.fillna(0.0)
    return X.fillna({c: safe_value(directions.get(c, 1)) for c in X.columns if not c.startswith("unk_")}).fillna(1.0)


def mask_channels(
    X: pd.DataFrame, channels: list[str] | tuple[str, ...], directions: dict[str, int] | None = None
) -> pd.DataFrame:
    """A channel entirely absent = no evidence from it (0) and fully unknown (1).

    FR-30 guarantee: every feature has a monotone constraint (direction learned on the training split; unk_*
    always non-increasing). A missing channel's features are set to the end of their range that can only lower
    the score (0 for increasing features, all of which are >= 0; a very large value for decreasing ones) and its
    unknown share to 1. Monotonicity then guarantees score(missing) <= score(full) for every session, and the
    calibration map is monotone too.
    """
    X = X.copy()
    drop = set(channels)
    for c in X.columns:
        chs = feature_channels(c)
        if c.startswith("unk_"):
            if chs[0] in drop:
                X[c] = 1.0
        elif chs and drop & set(chs):
            X[c] = safe_value((directions or {}).get(c, 1))
    return X


# plain-language templates (FR-24): feature family -> sentence builder
_PHRASES = {
    "phone": "a phone was detected in view",
    "multi_face": "more than one face was visible",
    "extra_person": "another person was detected in the room",
    "id_mismatch": "the face did not match the enrolment photo",
    "id_sim_low": "the face looked less like the enrolment photo",
    "pad_fail": "the liveness check failed",
    "no_face": "no face was visible",
    "gaze_off": "the candidate often looked away from the screen",
    "yaw_abs": "the head was turned away from the screen",
    "pitch_down": "the candidate often looked down",
    "paper": "paper or a book was visible",
    "foreign": "a voice other than the candidate's was heard",
    "vad": "there was speech in the room",
    "tab_hidden": "the exam tab was hidden",
    "fs_off": "the exam left full-screen mode",
    "paste_chars": "text was pasted into answers",
    "multi_monitor": "a second screen was connected",
    "keys_per_s": "typing activity was unusual",
    "reach": "the candidate reached out of frame",
    "hands_low": "hands were out of view",
    "phone_ring": "a phone ring was heard",
    "paper_rustle": "paper rustling was heard",
    "face_area": "the distance to the camera changed",
    "face_offset": "the face moved off-centre",
    "x_gaze_off_and_foreign": "the candidate looked away while another voice was heard",
    "x_phone_and_pitch_down": "a phone was visible while the candidate looked down",
    "x_multi_face_and_voice": "a second person was present while someone spoke",
    "x_mismatch_and_face": "a face that did not match enrolment was on camera",
    "x_gaze_off_no_typing": "the candidate looked away while not typing",
    "x_paper_and_pitch_down": "paper was visible while the candidate looked down",
    "x_paste_after_tab_return": "text was pasted shortly after returning to the exam tab",
    "x_foreign_and_keys": "the candidate typed while another voice was heard",
}


def explain_feature(name: str, direction: str) -> str:
    if name.startswith("unk_"):
        ch = name[4:].replace("_", " ")
        return f"Part of the {ch} signal was unavailable; missing data never counts against the candidate."
    base = name[3:] if name.startswith("bn_") else name
    key = base if base in _PHRASES else base.split("__")[0]
    phrase = _PHRASES.get(key, key.replace("_", " "))
    rel = " compared with the candidate's own first minute" if name.startswith("bn_") else ""
    if direction == "increases_risk":
        return f"Raised the score: {phrase}{rel}."
    return f"Lowered the score: little evidence that {phrase}{rel}."


@dataclass
class RiskModel:
    booster: Any
    features: list[str]
    meta: dict[str, Any]

    # ---- persistence ----------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> RiskModel:
        import lightgbm as lgb

        meta = json.loads((path / "bundle.json").read_text())
        return cls(lgb.Booster(model_file=str(path / "model.txt")), meta["features"], meta)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        (path / "bundle.json").write_text(json.dumps(self.meta, indent=2, default=float) + "\n")

    # ---- scoring --------------------------------------------------------------------------
    def raw(self, X: pd.DataFrame) -> np.ndarray:
        Xp = prepare(X.reindex(columns=self.features), self.meta.get("directions"))
        return np.asarray(self.booster.predict(Xp.to_numpy(dtype=float)), dtype=float)

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        c = self.meta["calibration"]
        return np.clip(np.interp(p, c["x"], c["y"]), 0.0, 1.0)

    def band(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self.meta["band"]
        return np.interp(p, b["x"], b["lo"]), np.interp(p, b["x"], b["hi"])

    def predict(self, X: pd.DataFrame, missing: list[str] | None = None) -> np.ndarray:
        """Calibrated risk. Missing channels are treated as 'no evidence' — never as suspicious (FR-30)."""
        if missing:
            X = mask_channels(X.reindex(columns=self.features), missing, self.meta.get("directions"))
        return self.calibrate(self.raw(X))

    def recommendation(self, risk: float) -> str:
        t = self.meta["tiers"]
        if risk >= t["PRIORITY_REVIEW"]:
            return "PRIORITY_REVIEW"
        if risk >= t["HUMAN_REVIEW"]:
            return "HUMAN_REVIEW"
        if risk >= t["ROUTINE_REVIEW"]:
            return "ROUTINE_REVIEW"
        return "NO_ACTION"

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP contributions (log-odds) from LightGBM; last column is the expected value."""
        Xp = prepare(X.reindex(columns=self.features), self.meta.get("directions"))
        return np.asarray(self.booster.predict(Xp.to_numpy(dtype=float), pred_contrib=True))

    def assess(
        self,
        session_id: str,
        feats: dict[str, float],
        flags: list[dict[str, Any]],
        channels_available: list[str],
        channels_missing: list[str],
    ) -> dict[str, Any]:
        X = pd.DataFrame([feats]).reindex(columns=self.features)
        risk = float(self.predict(X, channels_missing or None)[0])
        lo, hi = self.band(np.array([risk]))
        contrib = self.contributions(X)[0][:-1]
        top: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i in np.argsort(-np.abs(contrib)):  # top-5 contributors with distinct plain-language explanations
            direction = "increases_risk" if contrib[i] > 0 else "decreases_risk"
            text = explain_feature(self.features[i], direction)
            if text in seen:
                continue
            seen.add(text)
            top.append(
                {
                    "feature": self.features[i],
                    "shap": round(float(contrib[i]), 4),
                    "direction": direction,
                    "explanation": text,
                }
            )
            if len(top) == 5:
                break
        rec = self.recommendation(risk)
        if rec == "NO_ACTION":
            flags = []  # flags point a reviewer to moments to check; a session needing no review has none
        ra = {
            "schema": RISK_SCHEMA,
            "session_id": session_id,
            "overall_risk": round(risk, 4),
            "calibrated": True,
            "recommendation": rec,
            "confidence_band": [round(float(min(lo[0], risk)), 4), round(float(max(hi[0], risk)), 4)],
            "flags": flags,
            "top_contributors": top,
            "channels_available": channels_available,
            "channels_missing": channels_missing,
            "model_version": self.meta["model_version"],
            "feature_version": FEATURE_VERSION,
        }
        errs = risk_assessment_errors(ra)
        if errs:
            raise ValueError("; ".join(errs))
        return ra


def channels_from_features(feats: dict[str, float], threshold: float = 0.95) -> tuple[list[str], list[str]]:
    missing = [c for c in CHANNELS if feats.get(f"unk_{c}", 1.0) >= threshold]
    return [c for c in CHANNELS if c not in missing], missing
