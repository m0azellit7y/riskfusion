import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Badge, Empty, ErrorNotice, Loading, PageHeader, Panel, StatusBadge, useAsync } from "../components/ui";
import { api, type Page, type Session } from "../lib/api";
import { fmtDateTime, fmtDuration, LIGHTING, REC_LABEL, recTone, STATUS_LABEL, titleCase, WEBCAMS, WEBCAM_LABEL } from "../lib/format";

const PAGE = 50;

export default function SessionsPage() {
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const source = params.get("source") === "SIMULATED" ? "SIMULATED" : "MOCK";
  const status = params.get("status") ?? "";
  const lighting = params.get("lighting") ?? "";
  const webcam = params.get("webcam_class") ?? "";
  const violation = params.get("violation") ?? "";
  const offset = Number(params.get("offset") ?? 0);
  const [q, setQ] = useState(params.get("q") ?? "");

  const query = new URLSearchParams({ source, limit: String(PAGE), offset: String(offset) });
  if (status) query.set("status", status);
  if (lighting) query.set("lighting", lighting);
  if (webcam) query.set("webcam_class", webcam);
  if (violation) query.set("violation", violation);
  if (params.get("q")) query.set("q", params.get("q")!);
  const { data, error, loading, reload } = useAsync(() => api.get<Page<Session>>(`/sessions?${query}`), [query.toString()]);

  function set(k: string, v: string) {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    if (k !== "offset") next.delete("offset");
    setParams(next);
  }

  return (
    <>
      <PageHeader
        title="Sessions"
        description="Recorded mock sessions from consented volunteers, and simulated sessions from the generator."
        actions={<Link className="btn btn-primary" to="/sessions/new">New session</Link>}
      />
      <Panel flush>
        <div className="filters">
          <div className="segmented" role="group" aria-label="Source">
            <button aria-pressed={source === "MOCK"} onClick={() => setParams(new URLSearchParams({ source: "MOCK" }))}>Recorded</button>
            <button aria-pressed={source === "SIMULATED"} onClick={() => setParams(new URLSearchParams({ source: "SIMULATED" }))}>Simulated</button>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              set("q", q.trim());
            }}
          >
            <input className="input" placeholder={source === "MOCK" ? "Session ID or participant code" : "Session ID"} value={q}
              onChange={(e) => setQ(e.target.value)} aria-label="Search" />
          </form>
          {source === "MOCK" && (
            <select className="select" value={status} onChange={(e) => set("status", e.target.value)} aria-label="Status">
              <option value="">Any status</option>
              {(["CREATED", "CONSENTED", "RECORDING", "RECORDED", "UPLOADED", "PROCESSING", "COMPLETED", "REVIEWED", "FAILED", "DELETED"] as const).map((s) => (
                <option key={s} value={s}>{STATUS_LABEL[s]}</option>
              ))}
            </select>
          )}
          <select className="select" value={lighting} onChange={(e) => set("lighting", e.target.value)} aria-label="Lighting">
            <option value="">Any lighting</option>
            {LIGHTING.map((l) => <option key={l} value={l}>{titleCase(l)} light</option>)}
          </select>
          <select className="select" value={webcam} onChange={(e) => set("webcam_class", e.target.value)} aria-label="Webcam">
            <option value="">Any webcam</option>
            {WEBCAMS.map((w) => <option key={w} value={w}>{WEBCAM_LABEL[w]}</option>)}
          </select>
          <select className="select" value={violation} onChange={(e) => set("violation", e.target.value)} aria-label="Ground truth">
            <option value="">Any label</option>
            <option value="true">Violation</option>
            <option value="false">No violation</option>
          </select>
        </div>
        {loading && !data && <div style={{ padding: 16 }}><Loading /></div>}
        {error && <div style={{ padding: 16 }}><ErrorNotice message={error} onRetry={reload} /></div>}
        {data && data.items.length === 0 && (
          <Empty title="No sessions match" action={source === "MOCK" ? <Link className="btn btn-primary" to="/sessions/new">New session</Link> : undefined}>
            {source === "MOCK" ? "Change the filters, or record a new session." : "Change the filters, or generate and register a simulated dataset."}
          </Empty>
        )}
        {data && data.items.length > 0 && (
          <>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>{source === "MOCK" ? "Participant" : "Profile"}</th>
                    {source === "MOCK" && <th>Script</th>}
                    <th>Status</th>
                    <th>Conditions</th>
                    <th>Label</th>
                    <th>Risk</th>
                    {source === "SIMULATED" && <th>Split</th>}
                    <th className="num">Length</th>
                    {source === "MOCK" && <th>Created</th>}
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((s) => (
                    <tr key={s.id} className="clickable" onClick={() => nav(`/sessions/${s.id}`)}>
                      <td className="nowrap">{s.id}</td>
                      <td>{source === "MOCK" ? s.participant_code : titleCase(s.behavior_profile)}</td>
                      {source === "MOCK" && (
                        <td>{s.script_id}{s.is_rehearsal && <span className="faint small"> (rehearsal)</span>}</td>
                      )}
                      <td><StatusBadge status={s.status} /></td>
                      <td className="small muted nowrap">{titleCase(s.lighting)} light, {s.webcam_class?.toUpperCase()}</td>
                      <td>
                        {s.violation_label == null ? <span className="faint">—</span> : s.violation_label
                          ? <Badge tone="warn" plain>Violation</Badge> : <Badge plain>None</Badge>}
                      </td>
                      <td className="nowrap">
                        {s.recommendation ? (
                          <><span className="num">{s.risk?.toFixed(2)}</span>{" "}
                            <Badge tone={recTone(s.recommendation)} plain>{REC_LABEL[s.recommendation]}</Badge></>
                        ) : <span className="faint">—</span>}
                      </td>
                      {source === "SIMULATED" && <td>{titleCase(s.split)}</td>}
                      <td className="num">{fmtDuration(s.duration_s)}</td>
                      {source === "MOCK" && <td className="small nowrap">{fmtDateTime(s.created_at)}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span className="num">
                {offset + 1}–{Math.min(offset + PAGE, data.total)} of {data.total.toLocaleString()}
              </span>
              <div className="row">
                <button className="btn btn-sm" disabled={offset === 0} onClick={() => set("offset", String(Math.max(0, offset - PAGE)))}>Previous</button>
                <button className="btn btn-sm" disabled={offset + PAGE >= data.total} onClick={() => set("offset", String(offset + PAGE))}>Next</button>
              </div>
            </div>
          </>
        )}
      </Panel>
    </>
  );
}
