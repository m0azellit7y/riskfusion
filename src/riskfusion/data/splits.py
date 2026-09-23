"""Dataset splits and the sealed holdout (SRS DR-5, FR-21).

Rules implemented:

* Split **by session** (never by window) — the unit of assignment is a whole session.
* Split **by participant** when a ``participant_id`` column is present (mock data): every session
  of one person lands in the same split.
* Stratify by ``violation`` **and** ``behavior_profile``.
* 15 % **sealed holdout**, created once. Its session IDs are committed to the repository
  (``data/splits/<dataset_version>/``), and it can only be opened through
  :func:`open_sealed_holdout`, which appends to a committed access log and refuses a third access.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.60, "validation": 0.13, "calibration": 0.12, "holdout": 0.15}
MAX_HOLDOUT_ACCESSES = 2
ORDER = ("train", "validation", "calibration", "holdout")


class HoldoutSealedError(RuntimeError):
    """Raised on any attempt to read the sealed holdout outside the permitted accesses."""


def _allocate(n: int, fractions: dict[str, float]) -> list[str]:
    """Largest-remainder allocation of ``n`` items to splits, in ORDER."""
    raw = np.array([fractions[s] * n for s in ORDER])
    base = np.floor(raw).astype(int)
    rem = n - base.sum()
    for i in np.argsort(-(raw - base), kind="stable")[:rem]:
        base[i] += 1
    return [s for s, k in zip(ORDER, base, strict=True) for _ in range(k)]


def assign_splits(sessions: pd.DataFrame, seed: int, fractions: dict[str, float] = SPLIT_FRACTIONS) -> pd.DataFrame:
    """Return ``session_id, split`` for every session. Deterministic for a given seed."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    df = sessions[["session_id", "violation", "behavior_profile"]].copy()
    grouped = "participant_id" in sessions.columns and sessions["participant_id"].notna().any()
    if grouped:
        df["unit"] = sessions["participant_id"].fillna(sessions["session_id"]).astype(str)
    else:
        df["unit"] = df["session_id"].astype(str)
    # stratum of a unit: its most violating profile (ties broken by name for determinism)
    units = (
        df.sort_values(["unit", "violation", "behavior_profile"], ascending=[True, False, True])
        .groupby("unit", sort=True)
        .first()
    )
    units["stratum"] = units["violation"].astype(str) + "|" + units["behavior_profile"].astype(str)
    assignment: dict[str, str] = {}
    for _, grp in units.groupby("stratum", sort=True):
        ids = grp.index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        for uid, split in zip(ids, _allocate(len(ids), fractions), strict=True):
            assignment[uid] = split
    out = df[["session_id"]].copy()
    out["split"] = df["unit"].map(assignment)
    return out


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def split_dir(data_root: Path, dataset_version: str) -> Path:
    return data_root / "splits" / dataset_version


def create_splits(data_root: Path, dataset_version: str, sessions: pd.DataFrame, seed: int) -> dict[str, Any]:
    d = split_dir(data_root, dataset_version)
    if (d / "manifest.json").exists():
        raise HoldoutSealedError(
            f"splits for {dataset_version} already exist; the sealed holdout is created exactly once"
        )
    splits = assign_splits(sessions, seed)
    merged = splits.merge(sessions[["session_id", "violation", "behavior_profile"]], on="session_id")
    d.mkdir(parents=True, exist_ok=True)
    splits.sort_values("session_id").to_csv(d / "splits.csv", index=False)
    holdout_ids = sorted(splits.loc[splits["split"] == "holdout", "session_id"].astype(str))
    counts = merged.groupby(["split", "violation"]).size().rename("n").reset_index().to_dict(orient="records")
    manifest = {
        "split_version": f"splits-{dataset_version}-s{seed}",
        "dataset_version": dataset_version,
        "seed": seed,
        "fractions": SPLIT_FRACTIONS,
        "unit": "participant" if "participant_id" in sessions.columns else "session",
        "stratified_by": ["violation", "behavior_profile"],
        "counts": {s: int((splits["split"] == s).sum()) for s in ORDER},
        "counts_by_violation": counts,
        "positive_rate": {s: float(merged.loc[merged["split"] == s, "violation"].mean()) for s in ORDER},
        "holdout_ids_sha256": _ids_hash(holdout_ids),
        "holdout_session_ids": holdout_ids,
        "max_holdout_accesses": MAX_HOLDOUT_ACCESSES,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (d / "HOLDOUT_ACCESS_LOG.jsonl").touch()
    return manifest


def create_splits_for_dataset(data_root: Path, dataset_version: str, seed: int) -> dict[str, Any]:
    sessions = pd.read_parquet(data_root / "synthetic" / dataset_version / "sessions.parquet")
    return create_splits(data_root, dataset_version, sessions, seed)


def load_manifest(data_root: Path, dataset_version: str) -> dict[str, Any]:
    return dict(json.loads((split_dir(data_root, dataset_version) / "manifest.json").read_text()))


def load_split_ids(data_root: Path, dataset_version: str, split: str) -> list[str]:
    """IDs of a *non-holdout* split. The holdout is only reachable via open_sealed_holdout()."""
    if split == "holdout":
        raise HoldoutSealedError("the sealed holdout can only be opened with open_sealed_holdout()")
    if split not in ORDER:
        raise ValueError(f"unknown split {split!r}")
    df = pd.read_csv(split_dir(data_root, dataset_version) / "splits.csv")
    return sorted(df.loc[df["split"] == split, "session_id"].astype(str))


def holdout_accesses(data_root: Path, dataset_version: str) -> list[dict[str, Any]]:
    p = split_dir(data_root, dataset_version) / "HOLDOUT_ACCESS_LOG.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _git_head(cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def open_sealed_holdout(data_root: Path, dataset_version: str, purpose: str, operator: str | None = None) -> list[str]:
    """Open the sealed holdout. Logged; permitted at most MAX_HOLDOUT_ACCESSES times ever.

    The access log lives next to the committed manifest; committing it after each access is what
    makes the git history evidence the two accesses (FR-21).
    """
    if len(purpose.strip()) < 10:
        raise ValueError("state the purpose of this holdout access (>= 10 characters)")
    manifest = load_manifest(data_root, dataset_version)
    log = holdout_accesses(data_root, dataset_version)
    if len(log) >= MAX_HOLDOUT_ACCESSES:
        raise HoldoutSealedError(
            f"sealed holdout of {dataset_version} was already accessed {len(log)} times; access refused"
        )
    ids = list(manifest["holdout_session_ids"])
    if _ids_hash(ids) != manifest["holdout_ids_sha256"]:
        raise HoldoutSealedError("holdout manifest hash mismatch — the manifest was modified")
    entry = {
        "access_no": len(log) + 1,
        "purpose": purpose.strip(),
        "operator": operator or getpass.getuser(),
        "accessed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_head": _git_head(data_root),
    }
    with (split_dir(data_root, dataset_version) / "HOLDOUT_ACCESS_LOG.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return ids
