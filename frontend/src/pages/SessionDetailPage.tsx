import { useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AssessmentPanel } from "../components/AssessmentPanel";
import { Timeline, type Lane } from "../components/Timeline";
import { Badge, Empty, ErrorNotice, Loading, Modal, PageHeader, Panel, StatusBadge, useAsync, useToast } from "../components/ui";
import { API, api, type SessionDetail, type TelemetryEvent } from "../lib/api";
import { fmtBytes, fmtClock, fmtDate, fmtDateTime, fmtDuration, titleCase, VIOLATION_LABEL, WEBCAM_LABEL } from "../lib/format";

interface SignalTimeline {
  bin_ms: number;
  n_events: number;
  event_types: Record<string, number>;
  lanes: { label: string; intervals: { start_ms: number; end_ms: number }[] }[];
}

const TRUTH_COLOR = "#b42318";
const SIGNAL_COLOR = "#2c4a67";

function telemetryLanes(events: TelemetryEvent[], durMs: number): Lane[] {
  const hidden: { start: number; end: number }[] = [];
  const fsOff: { start: number; end: number }[] = [];
  let h: number | null = null;
  let f: number | null = null;
  for (const e of events) {
    if (e.event_type === "TAB_VISIBILITY") {
      if (e.payload.hidden && h == null) h = e.ts_ms;
      if (!e.payload.hidden && h != null) {
        hidden.push({ start: h, end: e.ts_ms });
        h = null;
      }
    }
    if (e.event_type === "FULLSCREEN_CHANGE") {
      if (!e.payload.active && f == null) f = e.ts_ms;
      if (e.payload.active && f != null) {
        fsOff.push({ start: f, end: e.ts_ms });
        f = null;
      }
    }
  }
  if (h != null) hidden.push({ start: h, end: durMs });
  if (f != null) fsOff.push({ start: f, end: durMs });
  const pastes = events.filter((e) => e.event_type === "PASTE").map((e) => ({ t: e.ts_ms, title: `Paste of ${e.payload.length} characters` }));
  const keys = events
    .filter((e) => e.event_type === "INPUT_ACTIVITY" && Number(e.payload.keystrokes) > 0)
    .map((e) => ({ start: e.ts_ms - Number(e.payload.window_s) * 1000, end: e.ts_ms, title: `${e.payload.keystrokes} keys` }));
  return [
    { label: "Tab hidden", color: SIGNAL_COLOR, intervals: hidden },
    { label: "Not full screen", color: "#8a96a3", intervals: fsOff },
    { label: "Paste", color: SIGNAL_COLOR, marks: pastes },
    { label: "Typing", color: "#9fb3c8", intervals: keys },
  ];
}

export default function SessionDetailPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const sess = useAsync(() => api.get<SessionDetail>(`/sessions/${id}`), [id]);
  const s = sess.data;
  const events = useAsync(
    () => (s && s.source === "MOCK" && Object.keys(s.event_counts).length ? api.get<TelemetryEvent[]>(`/sessions/${id}/events?limit=5000`) : Promise.resolve([])),
    [s?.id, s?.status],
  );
  const signals = useAsync(
    () => (s && s.source === "SIMULATED" ? api.get<SignalTimeline>(`/sessions/${id}/signal-timeline`) : Promise.resolve(null)),
    [s?.id],
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [busy, setBusy] = useState(false);

  const durMs = (s?.duration_s ?? 0) * 1000;
  const lanes = useMemo<Lane[]>(() => {
    if (!s) return [];
    const out: Lane[] = [];
    const ivs = s.label?.intervals ?? [];
    out.push({
      label: "Ground truth",
      color: TRUTH_COLOR,
      intervals: ivs.map((i) => ({ start: i.start_ms, end: i.end_ms, title: VIOLATION_LABEL[i.type] ?? i.type })),
    });
    if (s.source === "MOCK" && s.episodes.length) {
      out.push({
        label: "Cue shown",
        color: "#c98a2b",
        marks: s.episodes.filter((e) => e.shown_at_ms != null).map((e) => ({ t: e.shown_at_ms!, title: `Cue ${e.episode_no} shown` })),
      });
    }
    if (s.source === "MOCK" && events.data) out.push(...telemetryLanes(events.data, durMs));
    if (s.source === "SIMULATED" && signals.data) {
      out.push(...signals.data.lanes.map((l) => ({
        label: l.label,
        color: l.label === "Channel unknown" ? "#b8c2cc" : SIGNAL_COLOR,
        intervals: l.intervals.map((i) => ({ start: i.start_ms, end: i.end_ms })),
      })));
    }
    return out;
  }, [s, events.data, signals.data, durMs]);

  if (sess.loading && !s) return <Loading />;
  if (sess.error || !s) return <ErrorNotice message={sess.error ?? "Not found"} onRetry={sess.reload} />;

  const video = s.recordings.find((r) => r.kind === "webcam_av" && r.status === "STORED");
  const photo = s.recordings.find((r) => r.kind === "enrollment_image" && r.status === "STORED");
  const mock = s.source === "MOCK";

  async function confirmConsent() {
    setBusy(true);
    try {
      await api.post(`/sessions/${id}/confirm-consent`, { confirmed: true });
      nav(`/sessions/${id}/record`);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }
  async function del() {
    setBusy(true);
    try {
      await api.post(`/sessions/${id}/delete`, { confirm: true, reason: "deleted by operator" });
      toast("Session data deleted.");
      setDeleting(false);
      sess.reload();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  const actions = mock && (
    <>
      {s.status === "CREATED" && <button className="btn btn-primary" disabled={busy} onClick={confirmConsent}>Confirm consent and record</button>}
      {s.status === "CONSENTED" && <Link className="btn btn-primary" to={`/sessions/${id}/record`}>Record</Link>}
      {(s.status === "RECORDING" || s.status === "RECORDED" || s.status === "FAILED") && (
        <Link className="btn btn-primary" to={`/sessions/${id}/record`}>{s.status === "FAILED" ? "Record again" : "Open recorder"}</Link>
      )}
      {s.status !== "DELETED" && s.status !== "RECORDING" && (
        <button className="btn btn-danger" onClick={() => setDeleting(true)}>Delete session data</button>
      )}
    </>
  );

  return (
    <>
      <PageHeader
        crumbs={<Link to={`/sessions?source=${s.source}`}>{mock ? "Recorded sessions" : "Simulated sessions"}</Link>}
        title={s.id}
        description={
          <span className="row" style={{ gap: 10 }}>
            <StatusBadge status={s.status} />
            {mock ? (
              <span>
                {s.participant_id ? <Link to={`/participants/${s.participant_id}`}>{s.participant_code}</Link> : "—"}, script {s.script_id}
                {s.is_rehearsal && " (rehearsal)"}
              </span>
            ) : (
              <span>{titleCase(s.behavior_profile)} profile, {s.dataset_version}</span>
            )}
          </span>
        }
        actions={actions}
      />

      <div className="split-main">
        <div>
          {mock && (
            <Panel title="Recording" flush={!!video}>
              {video ? (
                <div className="video-frame" style={{ borderRadius: 0 }}>
                  <video
                    ref={videoRef}
                    controls
                    preload="metadata"
                    src={`${API}/recordings/${video.id}/content`}
                    onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime * 1000)}
                    aria-label="Session recording"
                  />
                </div>
              ) : s.status === "DELETED" ? (
                <p className="muted" style={{ margin: 0 }}>The recording was deleted.</p>
              ) : (
                <p className="muted" style={{ margin: 0 }}>No recording has been uploaded yet.</p>
              )}
            </Panel>
          )}
          <Panel
            title="Timeline"
            description={
              mock
                ? "Ground truth from the script, the cues as shown, and browser telemetry. Click to jump in the video."
                : "Ground truth against what the simulated detectors reported, in 10-second bins. Detector output is deliberately noisy."
            }
          >
            {durMs > 0 ? (
              <>
                {signals.loading && !mock && <Loading label="Reading events…" />}
                {signals.error && <div className="notice notice-info" style={{ marginBottom: 12 }}>{signals.error}</div>}
                <Timeline
                  durationMs={durMs}
                  lanes={lanes}
                  playheadMs={playhead}
                  onSeek={video ? (ms) => { if (videoRef.current) { videoRef.current.currentTime = ms / 1000; setPlayhead(ms); } } : undefined}
                />
                <div className="legend">
                  <span><i style={{ background: TRUTH_COLOR }} />Ground truth violation</span>
                  <span><i style={{ background: SIGNAL_COLOR }} />Observed signal</span>
                  {mock && <span><i style={{ background: "#c98a2b" }} />Script cue</span>}
                </div>
              </>
            ) : (
              <p className="muted" style={{ margin: 0 }}>The timeline appears once the session has been recorded.</p>
            )}
          </Panel>
          <AssessmentPanel session={s} onChange={sess.reload} />
          {mock && (
            <Panel title="Status history" flush>
              <div className="table-wrap">
              <table className="table">
                <tbody>
                  {s.history.map((h, i) => (
                    <tr key={i}>
                      <td className="nowrap small">{fmtDateTime(h.at)}</td>
                      <td>{titleCase(h.to_status)}</td>
                      <td className="small muted">{h.actor}{h.note ? `: ${h.note}` : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </Panel>
          )}
        </div>

        <div>
          <Panel title="Ground truth">
            {!mock && s.split === "holdout" ? (
              <p className="small muted" style={{ margin: 0 }}>
                Sealed holdout. The label and detector output are hidden until the planned evaluation.
              </p>
            ) : s.label ? (
              <>
                <div className="row" style={{ marginBottom: 10 }}>
                  {s.label.violation ? <Badge tone="warn">Violation</Badge> : <Badge>No violation</Badge>}
                  <Badge tone={s.label.confidence === "certain" ? "ok" : "warn"} plain>{titleCase(s.label.confidence)}</Badge>
                </div>
                {(s.label.intervals ?? []).length > 0 && (
                  <table className="table" style={{ marginBottom: 10 }}>
                    <tbody>
                      {s.label.intervals!.map((iv, i) => (
                        <tr key={i}>
                          <td className="small">{VIOLATION_LABEL[iv.type] ?? iv.type}</td>
                          <td className="num small">{fmtClock(iv.start_ms)}–{fmtClock(iv.end_ms)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                <p className="small muted" style={{ margin: 0 }}>
                  Source: {s.label.source}, {s.label.labeler}.
                  {s.label.confidence === "uncertain" && " Not every scheduled cue was shown, so the label is marked uncertain."}
                </p>
              </>
            ) : (
              <p className="muted small" style={{ margin: 0 }}>
                {mock ? "Created from the script when the recording is uploaded." : "No label."}
              </p>
            )}
          </Panel>
          <Panel title="Conditions">
            <dl className="dl">
              <dt>Lighting</dt><dd>{titleCase(s.lighting)}</dd>
              <dt>Webcam</dt><dd>{WEBCAM_LABEL[s.webcam_class ?? ""] ?? "—"}</dd>
              <dt>Room noise</dt><dd>{titleCase(s.room_noise)}</dd>
              <dt>Glasses</dt><dd>{s.eyewear == null ? "—" : s.eyewear ? "Yes" : "No"}</dd>
              <dt>Head covering</dt><dd>{s.head_covering == null ? "—" : s.head_covering ? "Yes" : "No"}</dd>
              {!mock && <><dt>Connection</dt><dd>{titleCase(s.connection_stability)}</dd></>}
              {!mock && <><dt>Movement</dt><dd className="num">{s.baseline_movement?.toFixed(2)}×</dd></>}
              {!mock && (s.channels_outage ?? []).length > 0 && <><dt>Channels missing</dt><dd>{s.channels_outage!.map(titleCase).join(", ")}</dd></>}
              <dt>Length</dt><dd>{fmtDuration(s.duration_s)}</dd>
              {!mock && <><dt>Split</dt><dd>{titleCase(s.split)}</dd></>}
            </dl>
          </Panel>
          {mock && (
            <Panel title="Stored files">
              {s.recordings.length === 0 ? (
                <p className="muted small" style={{ margin: 0 }}>None.</p>
              ) : (
                <div className="stack">
                  {s.recordings.map((r) => (
                    <div key={r.id} className="small">
                      <div className="row">
                        <strong>{r.kind === "webcam_av" ? "Video and audio" : "Enrolment photo"}</strong>
                        {r.status === "PURGED" ? <Badge plain>Deleted</Badge> : <span className="muted num">{fmtBytes(r.size_bytes)}</span>}
                      </div>
                      <div className="muted">SHA-256 <span className="code">{r.sha256.slice(0, 16)}…</span></div>
                      {r.status === "STORED" && r.retention_until && <div className="muted">Delete by {fmtDate(r.retention_until)}</div>}
                    </div>
                  ))}
                  {photo && <img src={`${API}/recordings/${photo.id}/content`} alt="Enrolment" style={{ maxWidth: 160, borderRadius: 6 }} />}
                </div>
              )}
            </Panel>
          )}
          <Panel title="Events">
            {mock ? (
              Object.keys(s.event_counts).length === 0 && s.dead_letter_count === 0 ? (
                <Empty title="No telemetry yet" />
              ) : (
                <dl className="dl">
                  {Object.entries(s.event_counts).map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}><dt>{titleCase(k)}</dt><dd className="num">{v}</dd></div>
                  ))}
                  <dt>Rejected (dead letter)</dt><dd className="num">{s.dead_letter_count}</dd>
                </dl>
              )
            ) : signals.data ? (
              <dl className="dl">
                <dt>Total</dt><dd className="num">{signals.data.n_events.toLocaleString()}</dd>
                {Object.entries(signals.data.event_types).slice(0, 10).map(([k, v]) => (
                  <div key={k} style={{ display: "contents" }}><dt>{titleCase(k)}</dt><dd className="num">{v.toLocaleString()}</dd></div>
                ))}
              </dl>
            ) : (
              <Loading />
            )}
          </Panel>
        </div>
      </div>

      {deleting && (
        <Modal
          title="Delete this session's data?"
          description="The recording, enrolment photo, telemetry and label are deleted permanently. The session remains listed as deleted for the audit trail."
          onClose={() => setDeleting(false)}
          footer={<><button className="btn" onClick={() => setDeleting(false)}>Cancel</button><button className="btn btn-danger-solid" disabled={busy} onClick={del}>Delete data</button></>}
        />
      )}
    </>
  );
}
