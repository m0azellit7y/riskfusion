import { Link, useNavigate } from "react-router-dom";
import { CoverageSummary } from "../components/Coverage";
import { Empty, ErrorNotice, Facts, Loading, PageHeader, Panel, StatusBadge, useAsync } from "../components/ui";
import { api, type Coverage, type SessionStatus } from "../lib/api";
import { fmtDateTime, fmtDuration, fmtInt, fmtPct } from "../lib/format";

interface Overview {
  counts: Record<string, Record<string, number>>;
  mock_needs_upload: number;
  mock_coverage: Coverage;
  recent_mock_sessions: {
    id: string;
    status: SessionStatus;
    script_id: string;
    created_at: string;
    participant_code: string;
    is_rehearsal: boolean;
    duration_s: number | null;
  }[];
  activity: { at: string; actor: string; action: string; entity_type: string; entity_id: string | null }[];
  datasets: { id: string; n_sessions: number; positive_rate: number; noise_source: string }[];
}

const ACTION_TEXT: Record<string, string> = {
  "participant.create": "registered a participant",
  "consent.sign": "recorded consent",
  "participant.withdraw": "processed a withdrawal",
  "session.create": "created a session",
  "session.status": "changed a session status",
  "recording.upload": "uploaded a recording",
  "dataset.register": "registered a dataset",
  "demographics.set": "saved optional details",
};

export default function OverviewPage() {
  const nav = useNavigate();
  const { data, error, loading, reload } = useAsync(() => api.get<Overview>("/overview"), []);
  if (loading && !data) return <Loading />;
  if (error || !data) return <ErrorNotice message={error ?? "No data"} onRetry={reload} />;

  const mock = data.counts.MOCK ?? {};
  const simulated = data.datasets.reduce((a, d) => a + d.n_sessions, 0);
  const cov = data.mock_coverage;

  return (
    <>
      <PageHeader
        title="Overview"
        description="Consented mock-session collection and the simulated datasets used to develop the risk models."
        actions={
          <Link className="btn btn-primary" to="/sessions/new">
            New session
          </Link>
        }
      />
      {data.mock_needs_upload > 0 && (
        <div className="notice notice-warn" style={{ marginBottom: 20 }}>
          <strong>
            {data.mock_needs_upload} {data.mock_needs_upload === 1 ? "recording is" : "recordings are"} waiting for upload.
          </strong>
          Open the session and upload it from the same browser before closing it, or the recording is lost.{" "}
          <Link to="/sessions?status=RECORDED">Show sessions</Link>
        </div>
      )}
      <Facts
        items={[
          { label: "Mock corpus", value: `${cov.eligible_sessions} / ${cov.target_sessions}`, note: "sessions toward FR-2" },
          { label: "Participants with consent", value: fmtInt(cov.active_participants), note: `${cov.participants_recorded} recorded` },
          { label: "Mock sessions, all states", value: fmtInt(Object.values(mock).reduce((a, b) => a + b, 0) - (mock.DELETED ?? 0)) },
          { label: "Simulated sessions", value: fmtInt(simulated), note: `${data.datasets.length} dataset version${data.datasets.length === 1 ? "" : "s"}` },
          { label: "Risk model", value: "Not trained", note: "arrives in Phase 4" },
        ]}
      />
      <div className="grid-2" style={{ marginTop: 20 }}>
        <Panel title="Mock corpus coverage" description="Sessions by lighting and webcam condition.">
          <CoverageSummary c={cov} />
        </Panel>
        <Panel
          title="Recent mock sessions"
          actions={
            <Link to="/sessions?source=MOCK" className="btn btn-sm">
              All sessions
            </Link>
          }
          flush
        >
          {data.recent_mock_sessions.length === 0 ? (
            <Empty title="No sessions recorded yet" action={<Link className="btn btn-primary" to="/sessions/new">Record the first session</Link>}>
              Register a participant, record their consent, then start a session.
            </Empty>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Participant</th>
                    <th>Script</th>
                    <th>Status</th>
                    <th className="num">Length</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_mock_sessions.map((s) => (
                    <tr key={s.id} className="clickable" onClick={() => nav(`/sessions/${s.id}`)}>
                      <td>{s.participant_code}</td>
                      <td>
                        {s.script_id}
                        {s.is_rehearsal && <span className="faint small"> (rehearsal)</span>}
                      </td>
                      <td>
                        <StatusBadge status={s.status} />
                      </td>
                      <td className="num">{fmtDuration(s.duration_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
        <Panel title="Simulated datasets" actions={<Link to="/datasets" className="btn btn-sm">Datasets</Link>} flush>
          {data.datasets.length === 0 ? (
            <Empty title="No simulated dataset registered">
              Generate one with <span className="code">make simulate</span>, then register it.
            </Empty>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th className="num">Sessions</th>
                  <th className="num">Violations</th>
                </tr>
              </thead>
              <tbody>
                {data.datasets.map((d) => (
                  <tr key={d.id}>
                    <td>{d.id}</td>
                    <td className="num">{fmtInt(d.n_sessions)}</td>
                    <td className="num">{fmtPct(d.positive_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel title="Recent activity" flush>
          {data.activity.length === 0 ? (
            <Empty title="Nothing has happened yet" />
          ) : (
            <table className="table">
              <tbody>
                {data.activity.map((a, i) => (
                  <tr key={i}>
                    <td className="small">
                      <strong>{a.actor}</strong> {ACTION_TEXT[a.action] ?? a.action}
                    </td>
                    <td className="small muted nowrap" style={{ textAlign: "right" }}>
                      {fmtDateTime(a.at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
    </>
  );
}
