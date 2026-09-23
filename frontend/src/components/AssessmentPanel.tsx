import { useEffect, useState } from "react";
import { API, api, type Assessment, type Job, type SessionDetail } from "../lib/api";
import { fmtClock, fmtDateTime, REC_LABEL, recTone, titleCase, VERDICT_LABEL } from "../lib/format";
import { Badge, Field, Panel, useAsync, useToast } from "./ui";

const VERDICTS = ["NO_CONCERN", "INCONCLUSIVE", "CONCERN_CONFIRMED"] as const;

export function AssessmentPanel({ session, onChange }: { session: SessionDetail; onChange: () => void }) {
  const toast = useToast();
  const sealed = session.source === "SIMULATED" && session.split === "holdout";
  const ra = useAsync(() => (sealed ? Promise.resolve(null) : api.get<Assessment | null>(`/sessions/${session.id}/assessment`)), [session.id, session.status]);
  const [job, setJob] = useState<Job | null>(null);
  const [verdict, setVerdict] = useState<string>("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  const canProcess = session.source === "MOCK" && ["UPLOADED", "FAILED", "COMPLETED", "REVIEWED"].includes(session.status) &&
    session.recordings.some((r) => r.kind === "webcam_av" && r.status === "STORED");

  useEffect(() => {
    if (session.source !== "MOCK") return;
    let alive = true;
    let timer = 0;
    const poll = async () => {
      const j = await api.get<Job | null>(`/sessions/${session.id}/job`).catch(() => null);
      if (!alive) return;
      setJob(j);
      if (j && (j.status === "QUEUED" || j.status === "RUNNING")) timer = window.setTimeout(poll, 1500);
      else if (j && session.status === "PROCESSING") onChange();
    };
    void poll();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [session.id, session.status, session.source, onChange]);

  async function analyze() {
    setBusy(true);
    try {
      await api.post(`/sessions/${session.id}/process`);
      toast("Analysis started. This takes about a minute per 30 minutes of video.");
      onChange();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }
  async function review() {
    setBusy(true);
    try {
      await api.post(`/sessions/${session.id}/review`, { verdict, note: note.trim() || null });
      toast("Review saved.");
      setVerdict("");
      setNote("");
      ra.reload();
      onChange();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  const running = job && (job.status === "QUEUED" || job.status === "RUNNING");
  const a = ra.data;
  return (
    <Panel
      title="Risk assessment"
      description="A recommendation for a human reviewer. It is never a finding of misconduct."
      actions={
        <>
          {a && <a className="btn btn-sm" href={`${API}/sessions/${session.id}/report`} target="_blank" rel="noreferrer">Reviewer report</a>}
          {canProcess && !running && (
            <button className="btn btn-sm btn-primary" disabled={busy} onClick={analyze}>
              {a ? "Analyse again" : "Analyse recording"}
            </button>
          )}
        </>
      }
    >
      {sealed && <p className="muted" style={{ margin: 0 }}>Sealed holdout: not scored in the dashboard.</p>}
      {running && (
        <div>
          <div className="row small" style={{ justifyContent: "space-between", marginBottom: 6 }}>
            <span>{job.stage ?? "Queued"}</span>
            <span className="num">{Math.round(job.progress * 100)}%</span>
          </div>
          <div className="progress"><span style={{ width: `${job.progress * 100}%` }} /></div>
        </div>
      )}
      {job?.status === "FAILED" && !running && (
        <div className="notice notice-bad" style={{ marginBottom: 12 }}>
          <strong>The last analysis failed.</strong>
          {job.message}
        </div>
      )}
      {!sealed && !running && !a && !ra.loading && (
        <p className="muted" style={{ margin: 0 }}>
          {session.source === "MOCK"
            ? canProcess ? "Not analysed yet. Analysing runs the detectors on the recording, then the risk model." : "Upload the recording first."
            : "No assessment stored for this session."}
        </p>
      )}
      {a && !running && (
        <div className="stack">
          <div className="row" style={{ gap: 14, alignItems: "baseline" }}>
            <span style={{ fontSize: 30, fontWeight: 600, color: "var(--ink)" }} className="num">{a.overall_risk.toFixed(2)}</span>
            <span className="small muted num">90% band {a.confidence_band[0].toFixed(2)}–{a.confidence_band[1].toFixed(2)}</span>
            <Badge tone={recTone(a.recommendation)}>{REC_LABEL[a.recommendation]}</Badge>
          </div>
          {session.source === "SIMULATED" && session.split === "train" && (
            <div className="notice notice-info small">
              This session was used to train the model, so its score is optimistic. Judge the model on validation
              and holdout results (Performance page).
            </div>
          )}
          <div>
            <h3 style={{ marginBottom: 6 }}>Why</h3>
            <ul style={{ margin: 0, paddingLeft: 18 }} className="small">
              {a.top_contributors.map((c) => (
                <li key={c.feature} style={{ marginBottom: 4 }}>
                  {c.explanation}
                </li>
              ))}
            </ul>
          </div>
          {a.flags.length > 0 && (
            <div>
              <h3 style={{ marginBottom: 6 }}>Moments to check</h3>
              <table className="table">
                <tbody>
                  {a.flags.slice(0, 12).map((f) => (
                    <tr key={f.flag_id}>
                      <td className="small nowrap num">{fmtClock(f.t_start_ms)}–{fmtClock(f.t_end_ms)}</td>
                      <td className="small">{f.explanation}</td>
                      <td className="small num muted">{f.confidence.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {a.flags.length > 12 && <p className="small muted">{a.flags.length - 12} more in the reviewer report.</p>}
            </div>
          )}
          {a.channels_missing.length > 0 && (
            <p className="small muted" style={{ margin: 0 }}>
              Unavailable: {a.channels_missing.map(titleCase).join(", ")}. Missing signals never raise the score.
            </p>
          )}
          <div>
            <h3 style={{ marginBottom: 6 }}>Reviewer decision</h3>
            {(a.reviews ?? []).map((r, i) => (
              <p key={i} className="small" style={{ margin: "0 0 4px" }}>
                <Badge plain tone={r.verdict === "CONCERN_CONFIRMED" ? "warn" : "neutral"}>{VERDICT_LABEL[r.verdict]}</Badge>{" "}
                {r.reviewer}, {fmtDateTime(r.created_at)}{r.note ? `: ${r.note}` : ""}
              </p>
            ))}
            <div className="segmented" role="group" aria-label="Verdict" style={{ margin: "6px 0 10px" }}>
              {VERDICTS.map((v) => (
                <button key={v} aria-pressed={verdict === v} onClick={() => setVerdict(v)}>{VERDICT_LABEL[v]}</button>
              ))}
            </div>
            <Field label="Note" htmlFor="rnote" hint="What you saw in the recording. Kept separate from the training labels.">
              <textarea id="rnote" className="textarea" rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
            </Field>
            <button className="btn btn-primary btn-sm" disabled={!verdict || busy} onClick={review}>Save decision</button>
          </div>
          <p className="small faint" style={{ margin: 0 }}>{a.model_version}, {a.feature_version}</p>
        </div>
      )}
    </Panel>
  );
}
