import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ConsentModal } from "../components/ConsentModal";
import { Badge, Empty, ErrorNotice, Field, Loading, Modal, PageHeader, Panel, StatusBadge, useAsync, useToast } from "../components/ui";
import { api, type Page, type ParticipantDetail, type Session } from "../lib/api";
import { fmtDate, fmtDateTime, fmtDuration } from "../lib/format";

const AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "prefer_not_to_say"];

export default function ParticipantDetailPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const p = useAsync(() => api.get<ParticipantDetail>(`/participants/${id}`), [id]);
  const sessions = useAsync(
    () => (p.data ? api.get<Page<Session>>(`/sessions?source=MOCK&include_deleted=true&q=${encodeURIComponent(p.data.code)}&limit=100`) : Promise.resolve(null)),
    [p.data?.code],
  );
  const [consentOpen, setConsentOpen] = useState(false);
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [demoOpen, setDemoOpen] = useState(false);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");

  if (p.loading && !p.data) return <Loading />;
  if (p.error || !p.data) return <ErrorNotice message={p.error ?? "Not found"} onRetry={p.reload} />;
  const d = p.data;

  async function withdraw() {
    setBusy(true);
    try {
      await api.post(`/participants/${id}/withdraw`, { confirm: true, reason: reason.trim() || null });
      toast(`${d.code} has withdrawn. Their recordings and derived data were deleted.`);
      setWithdrawOpen(false);
      p.reload();
      sessions.reload();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  async function saveDemographics() {
    setBusy(true);
    try {
      await api.put(`/participants/${id}/demographics`, { age_band: age || null, gender: gender.trim() || null });
      toast("Optional details saved.");
      setDemoOpen(false);
      p.reload();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  const active = d.status === "ACTIVE";
  return (
    <>
      <PageHeader
        crumbs={<Link to="/participants">Participants</Link>}
        title={d.code}
        description={`Registered ${fmtDate(d.created_at)}. Confirmed as 18 or older.`}
        actions={
          active && (
            <>
              {d.consent_active ? (
                <Link className="btn btn-primary" to={`/sessions/new?participant=${d.id}`}>
                  New session
                </Link>
              ) : (
                <button className="btn btn-primary" onClick={() => setConsentOpen(true)}>
                  Record consent
                </button>
              )}
              <button className="btn btn-danger" onClick={() => setWithdrawOpen(true)}>
                Withdraw participant
              </button>
            </>
          )
        }
      />
      {!active && (
        <div className="notice notice-info" style={{ marginBottom: 20 }}>
          <strong>Withdrawn {fmtDateTime(d.withdrawn_at)}.</strong>
          Recordings, derived data and optional details were deleted. The consent record is kept with the name removed.
        </div>
      )}
      <div className="split-main">
        <div>
          <Panel title="Sessions" flush>
            {sessions.loading && !sessions.data && <div style={{ padding: 16 }}><Loading /></div>}
            {sessions.data && sessions.data.items.length === 0 && (
              <Empty title="No sessions yet">{d.consent_active ? "Start a session when the participant is ready." : "Record consent first."}</Empty>
            )}
            {sessions.data && sessions.data.items.length > 0 && (
              <table className="table">
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>Script</th>
                    <th>Status</th>
                    <th className="num">Length</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.data.items.map((s) => (
                    <tr key={s.id} className="clickable" onClick={() => nav(`/sessions/${s.id}`)}>
                      <td>{s.id}</td>
                      <td>{s.script_id}</td>
                      <td><StatusBadge status={s.status} /></td>
                      <td className="num">{fmtDuration(s.duration_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
          <Panel title="Consent records" flush>
            {d.consents.length === 0 ? (
              <Empty title="No consent recorded" />
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>Signed</th>
                    <th>Form</th>
                    <th>Name</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {d.consents.map((c) => (
                    <tr key={c.id}>
                      <td>{fmtDateTime(c.signed_at)}</td>
                      <td>{c.consent_version}</td>
                      <td>{c.signed_name}</td>
                      <td>{c.withdrawn_at ? <Badge>Withdrawn {fmtDate(c.withdrawn_at)}</Badge> : <Badge tone="ok">Active</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
        </div>
        <div>
          <Panel
            title="Optional details"
            description="Voluntary. Stored separately and used only to check fairness, never as a model input."
            actions={active && <button className="btn btn-sm" onClick={() => setDemoOpen(true)}>{d.has_demographics ? "Update" : "Add"}</button>}
          >
            <p className="small muted" style={{ margin: 0 }}>
              {d.has_demographics ? "The participant has provided optional details." : "None provided."}
            </p>
          </Panel>
          <Panel title="Notes">
            <p className="small" style={{ margin: 0 }}>{d.notes || <span className="muted">No notes.</span>}</p>
          </Panel>
        </div>
      </div>

      {consentOpen && <ConsentModal participant={d} onClose={() => setConsentOpen(false)} onSigned={() => { setConsentOpen(false); p.reload(); }} />}
      {withdrawOpen && (
        <Modal
          title={`Withdraw ${d.code}`}
          description="This deletes every recording and all data derived from them, including sessions where this person was a helper. It cannot be undone."
          onClose={() => setWithdrawOpen(false)}
          footer={
            <>
              <button className="btn" onClick={() => setWithdrawOpen(false)}>Cancel</button>
              <button className="btn btn-danger-solid" disabled={typed !== d.code || busy} onClick={withdraw}>
                {busy ? "Deleting…" : "Withdraw and delete data"}
              </button>
            </>
          }
        >
          <Field label="Reason" htmlFor="wreason" hint="Optional. The participant does not have to give one.">
            <textarea id="wreason" className="textarea" value={reason} onChange={(e) => setReason(e.target.value)} />
          </Field>
          <Field label={`Type ${d.code} to confirm`} htmlFor="wconfirm">
            <input id="wconfirm" className="input" value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
          </Field>
        </Modal>
      )}
      {demoOpen && (
        <Modal
          title="Optional details"
          description="Only record what the participant chooses to share."
          onClose={() => setDemoOpen(false)}
          footer={
            <>
              <button className="btn" onClick={() => setDemoOpen(false)}>Cancel</button>
              <button className="btn btn-primary" disabled={busy} onClick={saveDemographics}>Save details</button>
            </>
          }
        >
          <Field label="Age band" htmlFor="age">
            <select id="age" className="select" value={age} onChange={(e) => setAge(e.target.value)}>
              <option value="">Not given</option>
              {AGE_BANDS.map((a) => (
                <option key={a} value={a}>{a === "prefer_not_to_say" ? "Prefer not to say" : a}</option>
              ))}
            </select>
          </Field>
          <Field label="Gender" htmlFor="gender" hint="Free text, in the participant's own words.">
            <input id="gender" className="input" value={gender} onChange={(e) => setGender(e.target.value)} maxLength={60} />
          </Field>
        </Modal>
      )}
    </>
  );
}
