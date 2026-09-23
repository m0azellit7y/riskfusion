import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ConsentModal } from "../components/ConsentModal";
import { Badge, Empty, ErrorNotice, Field, Loading, Modal, PageHeader, Panel, useAsync, useToast } from "../components/ui";
import { api, type Participant } from "../lib/api";
import { fmtDate } from "../lib/format";

export default function ParticipantsPage() {
  const nav = useNavigate();
  const toast = useToast();
  const { data, error, loading, reload } = useAsync(() => api.get<Participant[]>("/participants"), []);
  const [registering, setRegistering] = useState(false);
  const [adult, setAdult] = useState(false);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [consentFor, setConsentFor] = useState<Participant | null>(null);

  async function register() {
    setBusy(true);
    setFormError(null);
    try {
      const p = await api.post<Participant>("/participants", { adult_confirmed: adult, notes: notes.trim() || null });
      toast(`Registered ${p.code}.`);
      setRegistering(false);
      setAdult(false);
      setNotes("");
      reload();
      setConsentFor(p);
    } catch (e) {
      setFormError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Participants"
        description="Adults who volunteer for mock sessions. Each is identified by a code; names appear only on consent records."
        actions={
          <button className="btn btn-primary" onClick={() => setRegistering(true)}>
            Register participant
          </button>
        }
      />
      <Panel flush>
        {loading && !data && <div style={{ padding: 16 }}><Loading /></div>}
        {error && <div style={{ padding: 16 }}><ErrorNotice message={error} onRetry={reload} /></div>}
        {data && data.length === 0 && (
          <Empty title="No participants yet" action={<button className="btn btn-primary" onClick={() => setRegistering(true)}>Register participant</button>}>
            Register a volunteer, then take them through the consent form.
          </Empty>
        )}
        {data && data.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Consent</th>
                  <th className="num">Sessions</th>
                  <th>Registered</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.map((p) => (
                  <tr key={p.id} className="clickable" onClick={() => nav(`/participants/${p.id}`)}>
                    <td><strong>{p.code}</strong></td>
                    <td>
                      {p.status === "WITHDRAWN" ? (
                        <Badge tone="neutral">Withdrawn</Badge>
                      ) : p.consent_active ? (
                        <Badge tone="ok">Given ({p.consent_version})</Badge>
                      ) : (
                        <Badge tone="warn">Not recorded</Badge>
                      )}
                    </td>
                    <td className="num">{p.session_count}</td>
                    <td>{fmtDate(p.created_at)}</td>
                    <td style={{ textAlign: "right" }}>
                      {p.status === "ACTIVE" && !p.consent_active && (
                        <button
                          className="btn btn-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setConsentFor(p);
                          }}
                        >
                          Record consent
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {registering && (
        <Modal
          title="Register participant"
          description="A participant code is assigned automatically. You will take them through consent next."
          onClose={() => setRegistering(false)}
          footer={
            <>
              <button className="btn" onClick={() => setRegistering(false)}>Cancel</button>
              <button className="btn btn-primary" disabled={!adult || busy} onClick={register}>
                {busy ? "Registering…" : "Register"}
              </button>
            </>
          }
        >
          <label className="check">
            <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} />
            <span>
              I have confirmed that this participant is <strong>18 or older</strong>. Minors cannot take part.
            </span>
          </label>
          <Field label="Notes" htmlFor="pnotes" hint="Optional and internal. Do not enter the participant's name here.">
            <textarea id="pnotes" className="textarea" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} />
          </Field>
          {formError && <div className="notice notice-bad">{formError}</div>}
        </Modal>
      )}
      {consentFor && (
        <ConsentModal
          participant={consentFor}
          onClose={() => setConsentFor(null)}
          onSigned={() => {
            setConsentFor(null);
            reload();
          }}
        />
      )}
    </>
  );
}
