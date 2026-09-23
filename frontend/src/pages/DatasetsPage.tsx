import { useState } from "react";
import { Link } from "react-router-dom";
import { CoverageSummary } from "../components/Coverage";
import { Badge, Empty, ErrorNotice, Facts, Loading, PageHeader, Panel, useAsync } from "../components/ui";
import { api, type Coverage } from "../lib/api";
import { fmtDateTime, fmtDuration, fmtInt, fmtPct, titleCase, WEBCAM_LABEL } from "../lib/format";

interface Dataset {
  id: string;
  kind: string;
  config_version: string;
  config_hash: string;
  content_hash: string;
  seed: number;
  n_sessions: number;
  n_events: number;
  dead_letter_events: number;
  positive_rate: number;
  noise_source: string;
  generation_seconds: number;
  profile_counts: Record<string, number>;
  registered_at: string;
  splits: null | {
    split_version: string;
    counts: Record<string, number>;
    positive_rate: Record<string, number>;
    holdout_ids_sha256: string;
    holdout_accesses: { access_no: number; purpose: string; operator: string; accessed_at: string }[];
    max_holdout_accesses: number;
    created_at: string;
  };
}
type Slice = { value: string; sessions: number; positive_rate: number };
type Breakdown = Record<string, Slice[]> & { duration_s: { min: number; mean: number; max: number } };

const FACTOR_LABEL: Record<string, string> = {
  lighting: "Lighting",
  webcam_class: "Webcam",
  room_noise: "Room noise",
  connection_stability: "Connection",
  eyewear: "Glasses",
  head_covering: "Head covering",
};
const SPLITS = ["train", "validation", "calibration", "holdout"];

function sliceLabel(factor: string, v: string) {
  if (factor === "webcam_class") return WEBCAM_LABEL[v] ?? v;
  if (v === "True") return "Yes";
  if (v === "False") return "No";
  return titleCase(v);
}

function DatasetView({ d }: { d: Dataset }) {
  const bd = useAsync(() => api.get<Breakdown>(`/datasets/${d.id}/breakdown`), [d.id]);
  const maxProfile = Math.max(...Object.values(d.profile_counts ?? { x: 1 }));
  const [factor, setFactor] = useState("lighting");
  return (
    <div className="stack">
      <Facts
        items={[
          { label: "Sessions", value: fmtInt(d.n_sessions) },
          { label: "Events", value: fmtInt(d.n_events), note: `${fmtInt(d.dead_letter_events)} rejected` },
          { label: "Violation rate", value: fmtPct(d.positive_rate), note: "SRS expects 3–8%" },
          { label: "Generation time", value: fmtDuration(d.generation_seconds), note: "FR-1 limit: 10 min" },
        ]}
      />
      <div className="notice notice-warn">
        <strong>Detector error rates are estimates.</strong>
        The simulator uses the SRS's planning estimates (DR-3), made more pessimistic. They will be replaced by rates
        measured on the consented mock corpus (FR-6).
      </div>
      <div className="grid-2">
        <Panel title="Behaviour profiles" description="Sessions per profile. Violating profiles are highlighted.">
          {Object.entries(d.profile_counts ?? {}).map(([p, n]) => {
            const pos = !["clean", "fidgety_clean", "poor_environment"].includes(p);
            return (
              <div className="hbar" key={p}>
                <span>{titleCase(p)}</span>
                <span className="hbar-track"><span className={pos ? "pos" : ""} style={{ width: `${(n / maxProfile) * 100}%` }} /></span>
                <span className="num" style={{ textAlign: "right" }}>{fmtInt(n)}</span>
              </div>
            );
          })}
        </Panel>
        <Panel
          title="Condition slices"
          description="Sessions and violation rate per recording condition. These are the slices used for fairness evaluation."
          actions={
            <select className="select" value={factor} onChange={(e) => setFactor(e.target.value)} aria-label="Condition" style={{ width: "auto" }}>
              {Object.entries(FACTOR_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          }
          flush
        >
          {bd.loading && <div style={{ padding: 16 }}><Loading /></div>}
          {bd.error && <div style={{ padding: 16 }}><ErrorNotice message={bd.error} /></div>}
          {bd.data && (
            <table className="table">
              <thead><tr><th>{FACTOR_LABEL[factor]}</th><th className="num">Sessions</th><th className="num">Violation rate</th><th /></tr></thead>
              <tbody>
                {bd.data[factor].map((sl) => (
                  <tr key={sl.value}>
                    <td>{sliceLabel(factor, sl.value)}</td>
                    <td className="num">{fmtInt(sl.sessions)}</td>
                    <td className="num">{fmtPct(sl.positive_rate)}</td>
                    <td>{sl.sessions < 300 && <Badge tone="warn" plain>Under 300</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
      {d.splits ? (
        <div className="grid-2">
          <Panel title="Splits" description={`Assigned by session, stratified by violation and profile (${d.splits.split_version}).`} flush>
            <table className="table">
              <thead><tr><th>Split</th><th className="num">Sessions</th><th className="num">Violation rate</th></tr></thead>
              <tbody>
                {SPLITS.map((sp) => (
                  <tr key={sp}>
                    <td>
                      {sp === "holdout" ? "Sealed holdout" : titleCase(sp)}
                    </td>
                    <td className="num">{fmtInt(d.splits!.counts[sp])}</td>
                    <td className="num">{fmtPct(d.splits!.positive_rate[sp])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
          <Panel title="Sealed holdout" description="Opened at most twice: once in Phase 4, once in Phase 7. Every access is logged and committed.">
            <div className="row" style={{ marginBottom: 10 }}>
              <Badge tone={d.splits.holdout_accesses.length === 0 ? "ok" : "warn"}>
                {d.splits.holdout_accesses.length} of {d.splits.max_holdout_accesses} accesses used
              </Badge>
            </div>
            {d.splits.holdout_accesses.length > 0 ? (
              <table className="table">
                <tbody>
                  {d.splits.holdout_accesses.map((a) => (
                    <tr key={a.access_no}>
                      <td className="small nowrap">{fmtDateTime(a.accessed_at)}</td>
                      <td className="small">{a.purpose} <span className="muted">({a.operator})</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="small muted">Not opened yet.</p>
            )}
            <p className="small muted" style={{ margin: 0 }}>Manifest hash <span className="code">{d.splits.holdout_ids_sha256.slice(0, 24)}…</span></p>
          </Panel>
        </div>
      ) : (
        <div className="notice notice-info">No splits have been created for this dataset.</div>
      )}
      <Panel title="Provenance">
        <dl className="dl">
          <dt>Configuration</dt><dd>{d.config_version}, seed {d.seed}</dd>
          <dt>Configuration hash</dt><dd className="code">{d.config_hash}</dd>
          <dt>Content hash</dt><dd className="code">{d.content_hash}</dd>
          <dt>Registered</dt><dd>{fmtDateTime(d.registered_at)}</dd>
        </dl>
      </Panel>
    </div>
  );
}

export default function DatasetsPage() {
  const ds = useAsync(() => api.get<Dataset[]>("/datasets"), []);
  const cov = useAsync(() => api.get<Coverage>("/corpus/mock-coverage"), []);
  const [selected, setSelected] = useState<string | null>(null);
  const current = ds.data?.find((d) => d.id === selected) ?? ds.data?.[0];

  return (
    <>
      <PageHeader title="Datasets" description="The consented mock corpus and the simulated dataset versions." />
      <Panel title="Mock corpus" description="Consented recordings from volunteers (FR-2 target: 60 sessions across 3 lighting and 2 webcam conditions).">
        {cov.data ? (
          <div className="grid-2">
            <CoverageSummary c={cov.data} />
            <div>
              <h3 style={{ marginBottom: 8 }}>By script</h3>
              {Object.keys(cov.data.by_script).length === 0 ? (
                <p className="small muted">No eligible sessions yet. <Link to="/sessions/new">Record one</Link>.</p>
              ) : (
                <dl className="dl">
                  {Object.entries(cov.data.by_script).map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd className="num">{v}</dd></div>
                  ))}
                </dl>
              )}
            </div>
          </div>
        ) : cov.error ? <ErrorNotice message={cov.error} /> : <Loading />}
      </Panel>
      <div style={{ marginTop: 28 }}>
        <div className="row" style={{ marginBottom: 14 }}>
          <h2>Simulated datasets</h2>
          <span className="spacer" />
          {ds.data && ds.data.length > 1 && (
            <select className="select" style={{ width: "auto" }} value={current?.id} onChange={(e) => setSelected(e.target.value)} aria-label="Dataset version">
              {ds.data.map((d) => <option key={d.id} value={d.id}>{d.id}</option>)}
            </select>
          )}
        </div>
        {ds.loading && <Loading />}
        {ds.error && <ErrorNotice message={ds.error} onRetry={ds.reload} />}
        {ds.data && ds.data.length === 0 && (
          <Panel>
            <Empty title="No simulated dataset registered">
              Run <span className="code">make simulate</span> and <span className="code">make register</span>. See the README.
            </Empty>
          </Panel>
        )}
        {current && <DatasetView d={current} />}
      </div>
    </>
  );
}
