"""Session simulator (SRS DR-1, FR-1).

Each session is generated in three layers:

1. **Ground truth** — behaviour profile, nuisance factors (sampled independently of the label),
   violation episodes, and per-second true states (face present, second person, phone, gaze...).
2. **Detector noise model** — every simulated detector misses true states and fires bursty false
   positives at rates derived from the configuration (DR-3 planning estimates until FR-6 replaces
   them), modulated by nuisance factors (dim light, poor webcam, noisy room, eyewear...).
3. **Events** — detector output serialised as flat event.v1 rows, including explicit *_UNKNOWN
   events for connection gaps, unusable frames and missing channels (FR-7).

Every session draws from its own RNG stream ``default_rng([seed, index])`` so output is identical
regardless of sharding or generation order (seeded and reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from riskfusion.contracts import EVENT_TYPES, FLAT_COLUMNS, SLOT_MAP

from .config import PROFILE_VIOLATIONS, PROFILES, VIOLATION_TYPES, SimulatorConfig

Array = np.ndarray

# ---- vocabularies for compact columnar assembly --------------------------------------------
ET_NAMES: tuple[str, ...] = tuple(EVENT_TYPES)
ET_CODE = {n: i for i, n in enumerate(ET_NAMES)}
LABEL_VOCAB: tuple[str, ...] = tuple(
    sorted({a for et in EVENT_TYPES.values() for f in et.fields for a in (f.allowed or ())})
)
LABEL_CODE = {n: i for i, n in enumerate(LABEL_VOCAB)}
_DETECTOR_OF_CHANNEL = {
    "presence": "sim_face_detector",
    "identity": "sim_face_embedding",
    "liveness": "sim_pad",
    "attention": "sim_head_gaze",
    "environment": "sim_object_detector",
    "pose": "sim_body_pose",
    "audio_voice": "sim_vad_speaker",
    "audio_event": "sim_audio_tagger",
    "screen": "sim_browser_telemetry",
    "device": "sim_browser_telemetry",
    "behavioral": "sim_input_telemetry",
}
VISUAL_CHANNELS = ("presence", "identity", "liveness", "attention", "environment", "pose")
AUDIO_CHANNELS = ("audio_voice", "audio_event")


@dataclass
class _EventBuffer:
    ts: list[Array] = field(default_factory=list)
    et: list[Array] = field(default_factory=list)
    conf: list[Array] = field(default_factory=list)
    p: list[list[Array]] = field(default_factory=lambda: [[], [], [], []])
    label: list[Array] = field(default_factory=list)
    usable: list[Array] = field(default_factory=list)
    blur: list[Array] = field(default_factory=list)
    luma: list[Array] = field(default_factory=list)

    def add(
        self,
        event_type: str,
        ts_ms: Array,
        conf: Array | float,
        payload: dict[str, Array | float | str] | None = None,
        usable: Array | bool = True,
        blur: Array | float = np.nan,
        luma: Array | float = np.nan,
    ) -> None:
        n = len(ts_ms)
        if n == 0:
            return
        slots = SLOT_MAP[event_type]
        self.ts.append(np.asarray(ts_ms, dtype=np.int64))
        self.et.append(np.full(n, ET_CODE[event_type], dtype=np.int16))
        self.conf.append(np.broadcast_to(np.asarray(conf, dtype=np.float32), (n,)).copy())
        pv = [np.full(n, np.nan, dtype=np.float32) for _ in range(4)]
        lab = np.full(n, -1, dtype=np.int16)
        for name, value in (payload or {}).items():
            slot = slots[name]
            if slot == "label":
                if isinstance(value, str):
                    lab[:] = LABEL_CODE[value]
                else:
                    lab[:] = np.asarray(value, dtype=np.int16)
            else:
                pv[int(slot[1])] = np.broadcast_to(np.asarray(value, dtype=np.float32), (n,)).copy()
        for i in range(4):
            self.p[i].append(pv[i])
        self.label.append(lab)
        self.usable.append(np.broadcast_to(np.asarray(usable, dtype=bool), (n,)).copy())
        self.blur.append(np.broadcast_to(np.asarray(blur, dtype=np.float32), (n,)).copy())
        self.luma.append(np.broadcast_to(np.asarray(luma, dtype=np.float32), (n,)).copy())

    def unknown(self, channel: str, ts_ms: Array, reason: str) -> None:
        self.add(f"{channel.upper()}_UNKNOWN", ts_ms, 1.0, {"reason": reason}, usable=False)

    def arrays(self) -> dict[str, Array]:
        out = {
            "ts_ms": np.concatenate(self.ts),
            "et": np.concatenate(self.et),
            "confidence": np.concatenate(self.conf),
            "label": np.concatenate(self.label),
            "usable": np.concatenate(self.usable),
            "frame_blur": np.concatenate(self.blur),
            "frame_luma": np.concatenate(self.luma),
        }
        for i in range(4):
            out[f"p{i}"] = np.concatenate(self.p[i])
        order = np.lexsort((out["et"], out["ts_ms"]))
        return {k: v[order] for k, v in out.items()}


@dataclass
class SimulatedSession:
    meta: dict[str, Any]
    label: dict[str, Any]
    events: dict[str, Array]


# ---- helpers ------------------------------------------------------------------------------
def _choice(rng: np.random.Generator, dist: dict[str, float]) -> str:
    keys = list(dist)
    return str(keys[int(rng.choice(len(keys), p=np.array([dist[k] for k in keys])))])


def _intervals_to_mask(T: int, intervals: list[tuple[int, int]]) -> Array:
    m = np.zeros(T, dtype=bool)
    for s, e in intervals:
        m[max(0, s) : min(T, e)] = True
    return m


def _mask_to_intervals(mask: Array) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _random_intervals(
    rng: np.random.Generator, T: int, n: int, dur: tuple[float, float], lo: int = 0
) -> list[tuple[int, int]]:
    out = []
    for _ in range(n):
        d = int(rng.uniform(dur[0], dur[1]))
        d = max(1, min(d, T - lo - 1))
        s = int(rng.integers(lo, max(lo + 1, T - d)))
        out.append((s, s + d))
    return out


class _Noise:
    """Nuisance-modulated, bursty detector error model."""

    def __init__(self, cfg: SimulatorConfig, rng: np.random.Generator, nuisance: dict[str, Any]):
        self.cfg = cfg
        self.rng = rng
        self.mult: dict[str, list[float]] = {d: [1.0, 1.0] for d in cfg.detectors}
        eff = cfg.nuisance_effects
        for factor in ("lighting", "webcam_class", "room_noise"):
            level = nuisance[factor]
            for det, e in eff.get(factor, {}).get(level, {}).items():
                self.mult[det][0] *= e.fp
                self.mult[det][1] *= e.fn
        for flag in ("eyewear", "head_covering"):
            if nuisance[flag]:
                for det, e in eff.get(flag, {}).get("present", {}).items():
                    self.mult[det][0] *= e.fp
                    self.mult[det][1] *= e.fn

    def detect(self, truth: Array, detector: str, fp_scale: float = 1.0) -> Array:
        cfg, rng = self.cfg, self.rng
        n = truth.shape[0]
        fp_mult, fn_mult = self.mult[detector]
        p_miss = min(0.95, cfg.miss_prob(detector) * fn_mult)
        burst = cfg.detectors[detector].burst_s
        onset_p = min(0.5, cfg.fp_rate(detector) * fp_mult * fp_scale / burst)
        miss = truth & (rng.random(n) < p_miss)
        fp = np.zeros(n, dtype=bool)
        onsets = np.flatnonzero(~truth & (rng.random(n) < onset_p))
        if onsets.size:
            lengths = rng.geometric(1.0 / burst, size=onsets.size)
            for s, ln in zip(onsets, lengths, strict=True):
                fp[s : s + ln] = True
        return (truth & ~miss) | (fp & ~truth)


def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float = 0.9) -> Array:
    eps = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    out = np.empty(n)
    acc = rng.normal(0, sd)
    for i in range(n):  # n <= 5400; plain loop is ~1 ms
        acc = phi * acc + eps[i]
        out[i] = acc
    return out


# ---- the simulator ----------------------------------------------------------------------------
def simulate_session(cfg: SimulatorConfig, index: int, run_tag: str) -> SimulatedSession:
    rng = np.random.default_rng([cfg.seed, index])
    session_id = f"sim_{run_tag}_{index:06d}"

    # 1. profile, label, nuisance --------------------------------------------------------------
    weights = np.array([cfg.profiles[p].weight for p in PROFILES])
    profile = PROFILES[int(rng.choice(len(PROFILES), p=weights / weights.sum()))]
    if profile == "mixed":
        picks = rng.choice(len(VIOLATION_TYPES), size=2, replace=False)
        vtypes = tuple(sorted(VIOLATION_TYPES[int(i)] for i in picks))
    else:
        vtypes = PROFILE_VIOLATIONS.get(profile, ())
    nz = cfg.nuisance
    dm = nz.duration_min
    nuisance: dict[str, Any] = {
        "lighting": _choice(rng, nz.lighting),
        "webcam_class": _choice(rng, nz.webcam_class),
        "room_noise": _choice(rng, nz.room_noise),
        "connection_stability": _choice(rng, nz.connection_stability),
        "eyewear": bool(rng.random() < nz.eyewear_prob),
        "head_covering": bool(rng.random() < nz.head_covering_prob),
        "baseline_movement": float(
            rng.lognormal(nz.baseline_movement.lognormal_mean, nz.baseline_movement.lognormal_sigma)
        ),
    }
    duration_s = int(round(rng.triangular(dm.min, dm.mode, dm.max) * 60))
    T = duration_s
    t = np.arange(T)
    noise = _Noise(cfg, rng, nuisance)
    move = nuisance["baseline_movement"] * (1.6 if profile == "fidgety_clean" else 1.0)

    # 2. violation episodes (ground-truth intervals) --------------------------------------------
    episodes: dict[str, list[tuple[int, int]]] = {}
    for vt in vtypes:
        ep = cfg.episodes[vt]
        n_ep = int(rng.integers(ep.count[0], ep.count[1] + 1))
        if vt == "IMPERSONATION":
            start = 0 if rng.random() < 0.5 else int(rng.uniform(0.05, 0.4) * T)
            episodes[vt] = [(start, T)]
        else:
            ivs = _random_intervals(rng, T, n_ep, (ep.duration_s[0], ep.duration_s[1]), lo=30)
            episodes[vt] = _mask_to_intervals(_intervals_to_mask(T, ivs))
    gt = {vt: _intervals_to_mask(T, episodes.get(vt, [])) for vt in VIOLATION_TYPES}

    # 3. per-second true states -----------------------------------------------------------------
    away = _intervals_to_mask(T, _random_intervals(rng, T, int(rng.integers(0, 3)), (4, 25)))
    away &= ~gt["IMPERSONATION"]
    present = ~away
    second_person = gt["SECOND_PERSON"].copy()
    background_voice = np.zeros(T, dtype=bool)
    if profile == "poor_environment":  # people passing, TV, household noise — not violations
        second_person |= _intervals_to_mask(T, _random_intervals(rng, T, int(rng.integers(1, 6)), (2, 10)))
        background_voice |= rng.random(T) < rng.uniform(0.03, 0.10)
    second_voice_true = (gt["SECOND_PERSON"] & (rng.random(T) < 0.5)) | (
        gt["REMOTE_ASSISTANCE"] & (rng.random(T) < 0.6)
    )
    phone_true = gt["PHONE_USE"] & present
    paper_allowed = rng.random() < 0.3  # permitted scratch paper on desk
    paper_true = gt["NOTE_READING"] | (paper_allowed & (rng.random(T) < 0.05))
    glance_rate = {"clean": 0.01, "fidgety_clean": 0.04, "poor_environment": 0.025}.get(profile, 0.012)
    glances = _intervals_to_mask(T, _random_intervals(rng, T, int(glance_rate * T), (1, 4)))
    side = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    gaze_down = gt["NOTE_READING"] | gt["PHONE_USE"]
    gaze_side = gt["REMOTE_ASSISTANCE"] & (rng.random(T) < 0.7)
    gaze_off_true = (gaze_down | gaze_side | glances) & present
    tab_hidden_true = gt["TAB_SWITCH"].copy()
    if rng.random() < 0.08:  # accidental brief switch in a clean session
        tab_hidden_true |= _intervals_to_mask(T, _random_intervals(rng, T, 1, (1, 3)))
    multi_monitor_true = rng.random() < (0.4 if "TAB_SWITCH" in vtypes else 0.08)
    identity_mismatch_true = gt["IMPERSONATION"] & present
    spoof_true = identity_mismatch_true & (rng.random() < 0.25)
    reach_true = (
        (np.convolve(np.diff(np.concatenate(([0], phone_true.astype(int)))) == 1, [1, 1, 1])[:T] > 0)
        | (gt["NOTE_READING"] & (rng.random(T) < 0.1))
        | (rng.random(T) < 0.01)
    )
    typing = present & ~tab_hidden_true & (rng.random(T) < 0.45) & ~gaze_down

    # head pose (degrees)
    yaw = _ar1(rng, T, 5.0 * move) + np.where(glances, side * rng.uniform(22, 40, T), 0.0)
    yaw += np.where(gaze_side, side * 35.0, 0.0)
    pitch = -5.0 + _ar1(rng, T, 3.5 * move) + np.where(gaze_down, -28.0 + rng.normal(0, 4, T), 0.0)
    roll = _ar1(rng, T, 2.5 * move)

    # 4. observation quality, connection gaps, channel outages (FR-7) ----------------------------
    luma_base = {"bright": 150.0, "normal": 115.0, "dim": 52.0}[nuisance["lighting"]]
    blur_base = {"hd": 0.10, "sd": 0.20, "low": 0.34}[nuisance["webcam_class"]]
    luma = np.clip(luma_base + _ar1(rng, T, 12.0, 0.97), 0, 255)
    blur = np.clip(blur_base + np.abs(yaw - yaw.mean()) / 400 + rng.normal(0, 0.05, T), 0, 1)
    frame_usable = (luma > 32) & (blur < 0.6)
    cc = cfg.connection
    rate = cc.unstable_gaps_per_hour if nuisance["connection_stability"] == "unstable" else cc.stable_gaps_per_hour
    n_gaps = int(rng.poisson(rng.uniform(*rate) * T / 3600))
    gap = _intervals_to_mask(T, _random_intervals(rng, T, n_gaps, cc.gap_duration_s))
    outage = {c: bool(rng.random() < p) for c, p in cfg.channel_outage.items()}
    visual_out = outage.get("presence", False)
    audio_out = outage.get("audio_voice", False)

    buf = _EventBuffer()
    jit = rng.integers(0, 40, T)
    ts = t * 1000 + jit
    vis_ok = frame_usable & ~gap & (not visual_out)
    vis_reason = np.where(gap, "connection_lost", "low_light")

    def unknown_seconds(channel: str, sel: Array, reason_arr: Array | None, fixed: str | None = None) -> None:
        if not sel.any():
            return
        if fixed is not None:
            buf.unknown(channel, ts[sel], fixed)
            return
        assert reason_arr is not None
        for r in np.unique(reason_arr[sel]):
            m = sel & (reason_arr == r)
            buf.unknown(channel, ts[m], str(r))

    # presence -----------------------------------------------------------------------------------
    face_det = noise.detect(present, "face_present")
    extra_face = noise.detect(second_person & present, "face_count_gt1")
    face_count = face_det.astype(np.int64) + (extra_face & face_det).astype(np.int64)
    q_conf = np.clip(0.97 - blur * 0.5 - (luma < 70) * 0.15 + rng.normal(0, 0.02, T), 0.05, 1.0)
    area = np.where(face_det, np.clip(rng.normal(0.11, 0.02, T), 0.02, 0.5), 0.0)
    offx = np.where(face_det, np.clip(yaw / 90 + rng.normal(0, 0.03, T), -1, 1), 0.0)
    offy = np.where(face_det, np.clip(-pitch / 90 + rng.normal(0, 0.03, T), -1, 1), 0.0)
    if visual_out:
        for ch in VISUAL_CHANNELS:
            cadence = {"identity": 5, "liveness": 10, "environment": 2, "pose": 2}.get(ch, 1)
            sel = (t % cadence) == 0
            buf.unknown(ch, ts[sel] + {"identity": 200, "liveness": 300}.get(ch, 0), "device_unavailable")
    else:
        buf.add(
            "FACE_OBSERVATION",
            ts[vis_ok],
            q_conf[vis_ok],
            {
                "face_count": face_count[vis_ok],
                "bbox_area_ratio": area[vis_ok],
                "center_offset_x": offx[vis_ok],
                "center_offset_y": offy[vis_ok],
            },
            usable=True,
            blur=blur[vis_ok],
            luma=luma[vis_ok],
        )
        unknown_seconds("presence", ~vis_ok, vis_reason)

        # attention (needs a detected face)
        att_ok = vis_ok & face_det
        gaze_det = noise.detect(gaze_off_true, "gaze_off_screen", fp_scale=move)
        pose_sd = (
            3.0 * (1.5 if nuisance["webcam_class"] == "low" else 1.0) * (1.4 if nuisance["lighting"] == "dim" else 1.0)
        )
        prob = np.where(gaze_det, rng.normal(0.22, 0.12, T), rng.normal(0.86, 0.08, T))
        buf.add(
            "HEAD_GAZE",
            ts[att_ok] + 5,
            q_conf[att_ok] * 0.95,
            {
                "yaw": np.clip(yaw + rng.normal(0, pose_sd, T), -180, 180)[att_ok],
                "pitch": np.clip(pitch + rng.normal(0, pose_sd, T), -180, 180)[att_ok],
                "roll": np.clip(roll + rng.normal(0, pose_sd, T), -180, 180)[att_ok],
                "gaze_on_screen_prob": np.clip(prob, 0, 1)[att_ok],
            },
            blur=blur[att_ok],
            luma=luma[att_ok],
        )
        att_reason = np.where(vis_ok, "occluded", vis_reason)
        unknown_seconds("attention", ~att_ok, att_reason)

        # identity every 5 s
        chk = t % 5 == 0
        mism_det = noise.detect(identity_mismatch_true[chk], "identity_mismatch")
        sim_ok_mu = 0.70 - 0.08 * (nuisance["webcam_class"] == "low") - 0.06 * (nuisance["lighting"] == "dim")
        n_chk = int(chk.sum())
        sim = np.where(mism_det, rng.normal(0.22, 0.08, n_chk), rng.normal(sim_ok_mu, 0.06, n_chk))
        sim = np.clip(sim, -1, 1)
        fq = np.clip(1 - blur[chk] - (luma[chk] < 70) * 0.2, 0, 1)
        id_ok = (vis_ok & face_det)[chk]
        buf.add(
            "IDENTITY_CHECK",
            ts[chk][id_ok] + 200,
            np.clip(0.6 + 0.4 * fq, 0, 1)[id_ok],
            {"cosine_sim": sim[id_ok], "verified": (sim >= 0.45)[id_ok], "face_quality": fq[id_ok]},
            blur=blur[chk][id_ok],
            luma=luma[chk][id_ok],
        )
        unknown_seconds("identity", chk & ~(vis_ok & face_det), att_reason)

        # liveness every 10 s
        chk = t % 10 == 0
        spoof_det = noise.detect(spoof_true[chk], "liveness_spoof")
        score = np.clip(
            np.where(spoof_det, rng.normal(0.25, 0.1, int(chk.sum())), rng.normal(0.86, 0.07, int(chk.sum()))),
            0,
            1,
        )
        lv_ok = (vis_ok & face_det)[chk]
        buf.add(
            "LIVENESS_CHECK",
            ts[chk][lv_ok] + 300,
            np.full(int(lv_ok.sum()), 0.9, dtype=np.float32),
            {"pad_score": score[lv_ok], "pad_pass": (score >= 0.5)[lv_ok]},
        )
        unknown_seconds("liveness", chk & ~(vis_ok & face_det), att_reason)

        # environment: person count every 2 s, objects when detected
        chk = t % 2 == 0
        pc_det = noise.detect((second_person & present)[chk], "person_count_gt1")
        n_persons = np.where(present[chk] | pc_det, 1 + pc_det.astype(int), 0)
        env_ok = vis_ok[chk]
        buf.add("PERSON_COUNT", ts[chk][env_ok] + 100, 0.9, {"n_persons": n_persons[env_ok]})
        unknown_seconds("environment", chk & ~vis_ok, vis_reason)
        phone_det = noise.detect(phone_true, "phone") & vis_ok
        paper_det = noise.detect(paper_true, "book_paper") & vis_ok & ~phone_det
        for mask, lab in ((phone_det, "cell_phone"), (paper_det, "paper")):
            k = int(mask.sum())
            if k:
                buf.add(
                    "OBJECT_DETECTED",
                    ts[mask] + 120,
                    np.clip(rng.normal(0.62, 0.12, k), 0.3, 1.0),
                    {
                        "label": lab,
                        "bbox_x": np.clip(rng.normal(0.45, 0.1, k), 0, 0.9),
                        "bbox_y": np.clip(rng.normal(0.7, 0.08, k), 0, 0.9),
                        "bbox_w": np.clip(rng.normal(0.09, 0.02, k), 0.01, 0.1),
                        "bbox_h": np.clip(rng.normal(0.13, 0.03, k), 0.01, 0.1),
                    },
                )

        # pose every 2 s
        chk = t % 2 == 0
        reach_det = noise.detect(reach_true[chk], "reach_off_frame")
        hands = np.clip(np.where(typing[chk], 2, rng.integers(0, 3, int(chk.sum()))) - phone_true[chk], 0, 4)
        ps_ok = vis_ok[chk] & present[chk]
        buf.add(
            "BODY_POSE",
            ts[chk][ps_ok] + 150,
            0.8,
            {
                "hands_visible_count": hands[ps_ok],
                "reach_off_frame": reach_det[ps_ok],
                "torso_rotation": np.clip(yaw[chk] * 0.4 + rng.normal(0, 3, int(chk.sum())), -180, 180)[ps_ok],
            },
        )
        unknown_seconds("pose", chk & ~(vis_ok & present), np.where(vis_ok, "occluded", vis_reason))

    # audio ----------------------------------------------------------------------------------------
    aud_ok = ~gap & (not audio_out)
    if audio_out:
        buf.unknown("audio_voice", ts, "device_unavailable")
        buf.unknown("audio_event", ts[:1], "device_unavailable")
    else:
        foreign_det = noise.detect(second_voice_true | background_voice, "second_voice")
        cand_speech = rng.random(T) < 0.02
        noisy = nuisance["room_noise"] == "noisy"
        vad = cand_speech | foreign_det | (rng.random(T) < (0.06 if noisy else 0.015))
        buf.add(
            "VOICE_ACTIVITY",
            ts[aud_ok] + 10,
            np.clip(rng.normal(0.85, 0.06, T), 0, 1)[aud_ok],
            {
                "vad_active": vad[aud_ok],
                "speaker_is_candidate": (~foreign_det)[aud_ok],
                "n_speakers": np.where(vad, 1 + foreign_det.astype(int), 0)[aud_ok],
                "foreign_speech": foreign_det[aud_ok],
            },
        )
        unknown_seconds("audio_voice", ~aud_ok, None, fixed="connection_lost")
        ev_sel = {
            "keyboard_burst": typing & (rng.random(T) < 0.08),
            "paper_rustle": (paper_true & (rng.random(T) < 0.05)) | (rng.random(T) < (0.004 if noisy else 0.001)),
            "phone_ring": (np.diff(np.concatenate(([0], phone_true.astype(int)))) == 1) & (rng.random(T) < 0.2),
            "other": rng.random(T) < (0.01 if noisy else 0.002),
        }
        for lab, m in ev_sel.items():
            m = m & aud_ok
            if m.any():
                buf.add("AUDIO_EVENT", ts[m] + 30, np.clip(rng.normal(0.7, 0.1, int(m.sum())), 0.2, 1), {"label": lab})
        if gap.any():
            buf.unknown("audio_event", ts[gap][:1], "connection_lost")

    # screen / device / behavioural telemetry (browser-side, unaffected by network gaps) ----------
    if outage.get("screen", False):
        buf.unknown("screen", np.array([0]), "permission_denied")
    else:
        hidden = noise.detect(tab_hidden_true, "tab_hidden")
        change = np.flatnonzero(np.diff(np.concatenate(([0], hidden.astype(int)))) != 0)
        buf.add("TAB_VISIBILITY", np.array([0]), 1.0, {"hidden": False})
        if change.size:
            buf.add("TAB_VISIBILITY", ts[change] + 1, 1.0, {"hidden": hidden[change]})
        buf.add("FULLSCREEN_CHANGE", np.array([1]), 1.0, {"active": True})
        fs_exits = change[hidden[change]] if change.size else np.array([], dtype=int)
        if fs_exits.size:
            buf.add("FULLSCREEN_CHANGE", ts[fs_exits] + 2, 1.0, {"active": False})
            buf.add("FULLSCREEN_CHANGE", ts[np.minimum(fs_exits + 20, T - 1)] + 3, 1.0, {"active": True})
        ends = [e for _, e in episodes.get("TAB_SWITCH", []) if e < T and rng.random() < 0.6]
        own = list(np.flatnonzero(rng.random(T) < (0.0005)))
        if ends:
            buf.add("PASTE", ts[np.array(ends)] + 4, 1.0, {"length": rng.integers(60, 900, len(ends))})
        if own:
            buf.add("PASTE", ts[np.array(own)] + 4, 1.0, {"length": rng.integers(1, 40, len(own))})
    if outage.get("device", False):
        buf.unknown("device", np.array([0]), "not_observable")
    else:
        mm_det = noise.detect(np.array([multi_monitor_true]), "multi_monitor")
        buf.add("MONITOR_COUNT", np.array([0]), 1.0, {"count": 1 + int(mm_det[0])})
    chk = t % 10 == 9
    keys = rng.poisson(np.add.reduceat(typing.astype(float), np.arange(0, T, 10))[: int(chk.sum())] * 3.0)
    keys = keys + np.where(
        np.add.reduceat(gt["REMOTE_ASSISTANCE"].astype(float), np.arange(0, T, 10))[: int(chk.sum())] > 5,
        rng.poisson(25, int(chk.sum())),
        0,
    )
    buf.add("INPUT_ACTIVITY", ts[chk] + 50, 1.0, {"keystrokes": keys, "window_s": 10.0})

    events = buf.arrays()

    intervals = [
        {"start_ms": s * 1000, "end_ms": e * 1000, "type": vt}
        for vt in VIOLATION_TYPES
        for s, e in episodes.get(vt, [])
    ]
    violation = bool(cfg.profiles[profile].violation)
    meta = {
        "session_id": session_id,
        "behavior_profile": profile,
        "violation": violation,
        "violation_types": list(vtypes),
        "duration_s": duration_s,
        **nuisance,
        "channels_outage": sorted([c for c in ("presence", "audio_voice", "screen", "device") if outage.get(c, False)]),
        "n_events": int(events["ts_ms"].shape[0]),
    }
    label = {
        "session_id": session_id,
        "source": "simulated",
        "violation": violation,
        "violation_types": list(vtypes),
        "intervals": intervals,
        "labeler": f"simulator:{cfg.config_version}",
        "confidence": "certain",
    }
    return SimulatedSession(meta=meta, label=label, events=events)


def sessions_to_event_frame(sessions: list[SimulatedSession], detector_version: str) -> pd.DataFrame:
    """Assemble many simulated sessions into one flat event.v1 DataFrame (categorical columns)."""
    sids = [s.meta["session_id"] for s in sessions]
    lens = [s.events["ts_ms"].shape[0] for s in sessions]
    sess_codes = np.repeat(np.arange(len(sessions), dtype=np.int32), lens)
    cat = {k: np.concatenate([s.events[k] for s in sessions]) for k in sessions[0].events}
    et_codes = cat["et"].astype(np.int32)
    ch_of_et = np.array([EVENT_TYPES[n].channel for n in ET_NAMES], dtype=object)
    channels = pd.Categorical(ch_of_et[et_codes])
    detectors = pd.Categorical(np.array([_DETECTOR_OF_CHANNEL[c] for c in ch_of_et], dtype=object)[et_codes])
    df = pd.DataFrame(
        {
            "session_id": pd.Categorical.from_codes(sess_codes, categories=sids),
            "ts_ms": cat["ts_ms"],
            "channel": channels,
            "detector": detectors,
            "detector_version": pd.Categorical([detector_version] * len(et_codes)),
            "event_type": pd.Categorical.from_codes(et_codes, categories=list(ET_NAMES)),
            "confidence": cat["confidence"],
            "p0": cat["p0"],
            "p1": cat["p1"],
            "p2": cat["p2"],
            "p3": cat["p3"],
            "label": pd.Categorical.from_codes(cat["label"].astype(np.int32), categories=list(LABEL_VOCAB)),
            "usable": cat["usable"],
            "frame_blur": cat["frame_blur"],
            "frame_luma": cat["frame_luma"],
        }
    )
    return df[list(FLAT_COLUMNS)]
