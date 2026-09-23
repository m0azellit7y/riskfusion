import { useState } from "react";
import { api, type Participant } from "../lib/api";
import { ErrorNotice, Field, Loading, Markdown, Modal, useAsync, useToast } from "./ui";

const ITEMS = [
  ["consent_recording", "I agree to be recorded (video, sound, one reference photo and quiz-page activity) as described."],
  ["consent_analysis", "I agree to the recordings being analysed as described."],
  ["consent_retention", "I understand how long recordings are kept and when they are deleted."],
  ["understands_withdrawal", "I understand I can withdraw at any time and have my data deleted."],
] as const;

export function ConsentModal({
  participant,
  onClose,
  onSigned,
}: {
  participant: Participant;
  onClose: () => void;
  onSigned: () => void;
}) {
  const toast = useToast();
  const form = useAsync(() => api.get<{ version: string; text: string }>("/consent/current"), []);
  const [checks, setChecks] = useState<Record<string, boolean>>({});
  const [name, setName] = useState("");
  const [ethics, setEthics] = useState("");
  const [witness, setWitness] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const allChecked = ITEMS.every(([k]) => checks[k]);
  const canSign = allChecked && name.trim().length >= 2 && !busy && form.data;

  async function sign() {
    if (!form.data) return;
    setBusy(true);
    setError(null);
    try {
      await api.post(`/participants/${participant.id}/consent`, {
        consent_version: form.data.version,
        signed_name: name.trim(),
        ethics_reference: ethics.trim() || null,
        witnessed_by: witness.trim() || null,
        ...Object.fromEntries(ITEMS.map(([k]) => [k, !!checks[k]])),
      });
      toast(`Consent recorded for ${participant.code}.`);
      onSigned();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      wide
      title={`Consent for ${participant.code}`}
      description="Give the participant time to read the form. Every item must be agreed to before recording."
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={!canSign} onClick={sign}>
            {busy ? "Saving…" : "Record consent"}
          </button>
        </>
      }
    >
      {form.loading && <Loading />}
      {form.error && <ErrorNotice message={form.error} onRetry={form.reload} />}
      {form.data && (
        <>
          <div className="consent-text">
            <Markdown text={form.data.text} />
          </div>
          <div style={{ margin: "14px 0 6px" }}>
            {ITEMS.map(([k, text]) => (
              <label className="check" key={k}>
                <input type="checkbox" checked={!!checks[k]} onChange={(e) => setChecks({ ...checks, [k]: e.target.checked })} />
                <span>{text}</span>
              </label>
            ))}
          </div>
          <Field label="Participant's full name" htmlFor="signed_name" hint="Typed by the participant as their signature. Stored only on the consent record.">
            <input id="signed_name" className="input" value={name} onChange={(e) => setName(e.target.value)} autoComplete="off" />
          </Field>
          <div className="grid-2" style={{ gap: 12 }}>
            <Field label="Ethics approval reference" htmlFor="ethics" hint="Optional. Your institution's approval number (ETH-7).">
              <input id="ethics" className="input" value={ethics} onChange={(e) => setEthics(e.target.value)} />
            </Field>
            <Field label="Witnessed by" htmlFor="witness" hint="Optional.">
              <input id="witness" className="input" value={witness} onChange={(e) => setWitness(e.target.value)} />
            </Field>
          </div>
          <p className="small muted" style={{ margin: 0 }}>Form version {form.data.version}</p>
          {error && <div className="notice notice-bad" style={{ marginTop: 10 }}>{error}</div>}
        </>
      )}
    </Modal>
  );
}
