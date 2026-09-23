import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Steps } from "../components/Steps";
import { ErrorNotice, Field, Loading, PageHeader, Panel, useAsync, useToast } from "../components/ui";
import { api, type Participant, type ParticipantDetail, type ScriptsDoc, type SessionDetail } from "../lib/api";
import { LIGHTING, NOISE, WEBCAMS, WEBCAM_LABEL, fmtClock, fmtDateTime, titleCase, VIOLATION_LABEL } from "../lib/format";

function Segmented<T extends string>({ value, options, onChange, label, render }: {
  value: T; options: readonly T[]; onChange: (v: T) => void; label: string; render?: (v: T) => string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button type="button" key={o} aria-pressed={value === o} onClick={() => onChange(o)}>
          {render ? render(o) : titleCase(o)}
        </button>
      ))}
    </div>
  );
}

export default function NewSessionPage() {
  const nav = useNavigate();
  const toast = useToast();
  const [params] = useSearchParams();
  const participants = useAsync(() => api.get<Participant[]>("/participants"), []);
  const scripts = useAsync(() => api.get<ScriptsDoc>("/mock-scripts"), []);
  const [participantId, setParticipantId] = useState(params.get("participant") ?? "");
  const [helperId, setHelperId] = useState("");
  const [scriptId, setScriptId] = useState("clean");
  const [lighting, setLighting] = useState<(typeof LIGHTING)[number]>("normal");
  const [webcam, setWebcam] = useState<(typeof WEBCAMS)[number]>("hd");
  const [noise, setNoise] = useState<(typeof NOISE)[number]>("quiet");
  const [eyewear, setEyewear] = useState(false);
  const [headCovering, setHeadCovering] = useState(false);
  const [notes, setNotes] = useState("");
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [consent, setConsent] = useState<ParticipantDetail | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const eligible = useMemo(() => (participants.data ?? []).filter((p) => p.status === "ACTIVE" && p.consent_active), [participants.data]);
  const script = scripts.data?.scripts[scriptId];

  if (participants.loading || scripts.loading) return <Loading />;
  if (participants.error || scripts.error) return <ErrorNotice message={participants.error ?? scripts.error ?? ""} />;

  async function create() {
    setBusy(true);
    setError(null);
    try {
      const s = await api.post<SessionDetail>("/sessions", {
        participant_id: participantId,
        script_id: scriptId,
        helper_participant_id: script?.requires_helper ? helperId : null,
        lighting,
        webcam_class: webcam,
        room_noise: noise,
        eyewear,
        head_covering: headCovering,
        notes: notes.trim() || null,
      });
      setSession(s);
      setConsent(await api.get<ParticipantDetail>(`/participants/${participantId}`));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function confirmConsent() {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      await api.post(`/sessions/${session.id}/confirm-consent`, { confirmed: true });
      toast("Consent confirmed. Check the equipment next.");
      nav(`/sessions/${session.id}/record`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (session && consent) {
    const c = consent.consents.filter((x) => !x.withdrawn_at).at(-1);
    return (
      <>
        <PageHeader crumbs={<Link to="/sessions">Sessions</Link>} title="Confirm consent" description={`Session ${session.id} for ${consent.code}.`} />
        <Steps current={1} />
        <Panel title="Before recording">
          <dl className="dl" style={{ marginBottom: 16 }}>
            <dt>Participant</dt>
            <dd>{consent.code}</dd>
            <dt>Consent form</dt>
            <dd>{c ? `${c.consent_version}, signed ${fmtDateTime(c.signed_at)}` : "—"}</dd>
            <dt>Script</dt>
            <dd>{script?.title}</dd>
          </dl>
          {script?.requires_helper && (
            <div className="notice notice-info" style={{ marginBottom: 12 }}>
              This script involves a helper. Their consent is on file; make sure they are present and have read the form too.
            </div>
          )}
          <label className="check">
            <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
            <span>The participant has been reminded what will be recorded, that they can stop at any time, and agrees to continue today.</span>
          </label>
          {error && <div className="notice notice-bad" style={{ marginTop: 10 }}>{error}</div>}
          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn btn-primary" disabled={!confirmed || busy} onClick={confirmConsent}>
              Confirm and continue
            </button>
            <Link className="btn" to={`/sessions/${session.id}`}>Not now</Link>
          </div>
        </Panel>
      </>
    );
  }

  const canCreate = participantId && script && (!script.requires_helper || (helperId && helperId !== participantId));
  return (
    <>
      <PageHeader
        crumbs={<Link to="/sessions">Sessions</Link>}
        title="New session"
        description="Choose the participant, the script they will follow, and describe the recording conditions."
      />
      <Steps current={0} />
      {eligible.length === 0 ? (
        <Panel>
          <div className="empty">
            <h3>No participant is ready to record</h3>
            <p>A session needs a registered adult participant with signed consent.</p>
            <Link className="btn btn-primary" to="/participants">Go to participants</Link>
          </div>
        </Panel>
      ) : (
        <div className="split-main">
          <div>
            <Panel title="Participant and script">
              <Field label="Participant" htmlFor="participant" hint="Only participants with active consent are listed.">
                <select id="participant" className="select" value={participantId} onChange={(e) => setParticipantId(e.target.value)}>
                  <option value="">Choose a participant</option>
                  {eligible.map((p) => (
                    <option key={p.id} value={p.id}>{p.code}</option>
                  ))}
                </select>
              </Field>
              <div className="field">
                <span className="label">Script</span>
                <div className="choice-list" role="radiogroup" aria-label="Script">
                  {Object.entries(scripts.data!.scripts).map(([id, s]) => (
                    <label key={id} className={`choice ${scriptId === id ? "selected" : ""}`}>
                      <input type="radio" name="script" checked={scriptId === id} onChange={() => setScriptId(id)} />
                      <span>
                        <span className="choice-title">{s.title}</span>
                        <span className="choice-desc" style={{ display: "block" }}>{s.summary}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              {script?.requires_helper && (
                <Field label="Helper" htmlFor="helper" hint="A second consenting adult who appears in or speaks during the session.">
                  <select id="helper" className="select" value={helperId} onChange={(e) => setHelperId(e.target.value)}>
                    <option value="">Choose the helper</option>
                    {eligible.filter((p) => p.id !== participantId).map((p) => (
                      <option key={p.id} value={p.id}>{p.code}</option>
                    ))}
                  </select>
                </Field>
              )}
            </Panel>
            <Panel title="Recording conditions" description="Describe the room as it is. These become the fairness slices in evaluation.">
              <div className="field">
                <span className="label">Lighting</span>
                <Segmented label="Lighting" value={lighting} options={LIGHTING} onChange={setLighting} />
              </div>
              <div className="field">
                <span className="label">Webcam quality</span>
                <Segmented label="Webcam quality" value={webcam} options={WEBCAMS} onChange={setWebcam} render={(w) => WEBCAM_LABEL[w]} />
              </div>
              <div className="field">
                <span className="label">Room noise</span>
                <Segmented label="Room noise" value={noise} options={NOISE} onChange={setNoise} />
              </div>
              <label className="check"><input type="checkbox" checked={eyewear} onChange={(e) => setEyewear(e.target.checked)} /> Participant wears glasses</label>
              <label className="check"><input type="checkbox" checked={headCovering} onChange={(e) => setHeadCovering(e.target.checked)} /> Participant wears a head covering</label>
              <Field label="Notes" htmlFor="snotes">
                <textarea id="snotes" className="textarea" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} />
              </Field>
            </Panel>
          </div>
          <div>
            <Panel title="What will happen">
              {script && (
                <>
                  <p className="small">{script.summary}</p>
                  {script.episodes.length === 0 ? (
                    <p className="small muted">No cues. The participant answers the quiz normally.</p>
                  ) : (
                    <table className="table">
                      <thead>
                        <tr><th>At</th><th>Cue</th><th className="num">For</th></tr>
                      </thead>
                      <tbody>
                        {script.episodes.map((e, i) => (
                          <tr key={i}>
                            <td className="num">{fmtClock(e.start_s * 1000)}</td>
                            <td className="small">{VIOLATION_LABEL[e.type] ?? e.type}</td>
                            <td className="num">{e.duration_s < 0 ? "rest" : `${e.duration_s} s`}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <p className="small muted" style={{ marginTop: 12, marginBottom: 0 }}>
                    {script.rehearsal
                      ? "Rehearsal: not added to the dataset. Use it to test the camera, microphone and upload."
                      : `Record for ${scripts.data!.min_duration_s / 60} to ${scripts.data!.max_duration_s / 60} minutes. The ground-truth label comes from this script.`}
                  </p>
                </>
              )}
            </Panel>
            {error && <div className="notice notice-bad">{error}</div>}
            <div className="row" style={{ marginTop: 16 }}>
              <button className="btn btn-primary btn-lg" disabled={!canCreate || busy} onClick={create}>
                {busy ? "Creating…" : "Create session"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
