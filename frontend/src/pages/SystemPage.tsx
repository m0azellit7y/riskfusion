import { Badge, ErrorNotice, Loading, PageHeader, Panel, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtDateTime, fmtInt, titleCase } from "../lib/format";

interface Status {
  health: { status: string; version: string; contracts: string[]; checks: Record<string, string> };
  schema_revision: string;
  row_counts: Record<string, number>;
  storage: { backend: string; used_bytes: number; stored_recordings: number };
  channels: string[];
  consent_version: string;
  pipeline: { stage: string; state: "available" | "not_built" }[];
}
interface AuditRow { at: string; actor: string; action: string; entity_type: string; entity_id: string | null; details: Record<string, unknown> | null }

export default function SystemPage() {
  const st = useAsync(() => api.get<Status>("/system/status"), []);
  const audit = useAsync(() => api.get<AuditRow[]>("/audit?limit=50"), []);
  if (st.loading && !st.data) return <Loading />;
  if (st.error || !st.data) return <ErrorNotice message={st.error ?? ""} onRetry={st.reload} />;
  const d = st.data;
  return (
    <>
      <PageHeader
        title="System status"
        description="Service health, what parts of the pipeline exist in this build, and the audit trail."
        actions={<button className="btn" onClick={() => { st.reload(); audit.reload(); }}>Refresh</button>}
      />
      <div className="grid-3">
        <Panel title="Service">
          <dl className="dl">
            <dt>API</dt><dd><Badge tone={d.health.status === "ok" ? "ok" : "bad"}>{d.health.status === "ok" ? "Healthy" : "Degraded"}</Badge></dd>
            {Object.entries(d.health.checks).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}><dt>{titleCase(k)}</dt><dd>{v === "ok" ? "Connected" : v}</dd></div>
            ))}
            <dt>Version</dt><dd>{d.health.version}</dd>
            <dt>Schema revision</dt><dd>{d.schema_revision}</dd>
            <dt>Contracts</dt><dd>{d.health.contracts.join(", ")}</dd>
            <dt>Consent form</dt><dd>{d.consent_version}</dd>
          </dl>
        </Panel>
        <Panel title="Storage">
          <dl className="dl">
            <dt>Backend</dt><dd>{titleCase(d.storage.backend)} files</dd>
            <dt>Media stored</dt><dd className="num">{fmtBytes(d.storage.used_bytes)}</dd>
            <dt>Recordings kept</dt><dd className="num">{d.storage.stored_recordings}</dd>
          </dl>
        </Panel>
        <Panel title="Records">
          <dl className="dl">
            {Object.entries(d.row_counts).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}><dt>{titleCase(k)}</dt><dd className="num">{fmtInt(v)}</dd></div>
            ))}
          </dl>
        </Panel>
      </div>
      <Panel title="Pipeline" description="Stages marked not built are planned for later phases and are not shown elsewhere in the product.">
        <table className="table">
          <tbody>
            {d.pipeline.map((p) => (
              <tr key={p.stage}>
                <td>{p.stage}</td>
                <td style={{ textAlign: "right" }}>{p.state === "available" ? <Badge tone="ok">Available</Badge> : <Badge>Not built yet</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <Panel title="Audit log" description="Every consent, status change, upload, withdrawal and deletion, most recent first." flush>
        {audit.data && (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>When</th><th>Who</th><th>Action</th><th>Record</th></tr></thead>
              <tbody>
                {audit.data.map((a, i) => (
                  <tr key={i}>
                    <td className="small nowrap">{fmtDateTime(a.at)}</td>
                    <td className="small">{a.actor}</td>
                    <td className="small">{a.action}{a.details && "to" in a.details ? `: ${String(a.details.to).toLowerCase()}` : ""}</td>
                    <td className="small muted">{a.entity_type} {a.entity_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}
