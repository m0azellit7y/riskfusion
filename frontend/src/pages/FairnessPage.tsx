import { useState } from "react";
import { IntervalRows } from "../components/Charts";
import { Badge, ErrorNotice, Loading, PageHeader, Panel, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { fmtPct, titleCase, WEBCAM_LABEL } from "../lib/format";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;
const FACTORS: Record<string, string> = { lighting: "Lighting", webcam_class: "Webcam", room_noise: "Room noise",
  connection_stability: "Connection", eyewear: "Glasses", head_covering: "Head covering" };
const lvl = (f: string, v: string) => (f === "webcam_class" ? WEBCAM_LABEL[v] ?? v : v === "True" ? "Yes" : v === "False" ? "No" : titleCase(v));

export default function FairnessPage() {
  const ev = useAsync(() => api.get<Any>("/model/evaluation"), []);
  const [factor, setFactor] = useState("lighting");
  if (ev.loading) return <Loading />;
  if (ev.error || !ev.data?.evaluation) return <ErrorNotice message={ev.error ?? "No evaluation available"} onRetry={ev.reload} />;
  const e = ev.data.evaluation;
  const s = e.slices[factor];
  const nr = ev.data.noise_robustness;
  const drift = ev.data.drift;
  const mit = e.mitigation.results;
  return (
    <>
      <PageHeader title="Fairness and monitoring"
        description="How the false-positive rate — the rate of clean sessions sent to a reviewer — varies by recording condition, how the model behaves when signals are missing or noisy, and whether new data has drifted." />
      <Panel title="False-positive rate by condition"
        description="Validation split, at the human-review threshold. Bars show 95% bootstrap intervals; slices under 300 sessions are underpowered."
        actions={<select className="select" style={{ width: "auto" }} value={factor} onChange={(x) => setFactor(x.target.value)} aria-label="Condition">
          {Object.entries(FACTORS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>}>
        <div className="row" style={{ marginBottom: 10 }}>
          <span>FPR disparity (highest ÷ lowest): <strong className="num">{s.fpr_disparity.toFixed(2)}</strong></span>
          <span className="small muted num">95% CI {s.fpr_disparity_ci.map((x: number) => x.toFixed(2)).join("–")}</span>
          <Badge tone={s.fpr_disparity < 1.3 ? "ok" : "warn"}>{s.fpr_disparity < 1.3 ? "Within the 1.3 target" : "Above the 1.3 target"}</Badge>
        </div>
        <IntervalRows max={Math.max(0.12, ...s.levels.map((l: Any) => l.fpr_ci[1] || 0))} reference={{ value: 0.03, label: "3% budget" }}
          rows={s.levels.map((l: Any) => ({ label: `${lvl(factor, l.level)} (${l.n})`, value: l.fpr, lo: l.fpr_ci[0], hi: l.fpr_ci[1] }))} />
        <table className="table" style={{ marginTop: 8 }}>
          <thead><tr><th>{FACTORS[factor]}</th><th className="num">Sessions</th><th className="num">Violations</th><th className="num">Recall</th><th className="num">ECE</th><th /></tr></thead>
          <tbody>{s.levels.map((l: Any) => (
            <tr key={l.level}><td>{lvl(factor, l.level)}</td><td className="num">{l.n}</td><td className="num">{l.positives}</td>
              <td className="num">{Number.isFinite(l.recall) ? fmtPct(l.recall) : "—"}</td><td className="num">{l.ece.toFixed(3)}</td>
              <td>{l.underpowered && <Badge plain tone="warn">Under 300</Badge>}</td></tr>))}
          </tbody>
        </table>
      </Panel>
      <div className="grid-2" style={{ marginTop: 20 }}>
        <Panel title="Mitigation" description={e.mitigation.method}>
          <table className="table">
            <thead><tr><th /><th className="num">Before</th><th className="num">After</th></tr></thead>
            <tbody>
              {["val_pr_auc", "recall", "fpr", "fpr_disparity_lighting", "fpr_disparity_webcam_class", "fpr_disparity_room_noise", "fpr_disparity_head_covering"].map((k) => {
                const [b, a] = [mit["before (unweighted)"][k], mit["after (slice-balanced weights)"][k]];
                const pct = ["recall", "fpr"].includes(k);
                return <tr key={k}><td className="small">{k.replace("val_", "").replace("fpr_disparity_", "FPR disparity, ").replace(/_/g, " ")}</td>
                  <td className="num">{pct ? fmtPct(b) : b.toFixed(2)}</td><td className="num">{pct ? fmtPct(a) : a.toFixed(2)}</td></tr>;
              })}
            </tbody>
          </table>
          <p className="small muted" style={{ marginBottom: 0 }}>Not adopted for the deployed model: it lowered overall FPR but widened disparity for lighting and head covering. See the model report.</p>
        </Panel>
        <Panel title="Missing signals" description="Each channel removed entirely at inference (validation). Missing data must never raise a score.">
          <table className="table">
            <thead><tr><th>Channel removed</th><th className="num">Scores raised</th><th className="num">Recall</th></tr></thead>
            <tbody>{e.missing_channel.map((r: Any) => (
              <tr key={r.channel}><td>{titleCase(r.channel)}</td>
                <td className="num"><Badge plain tone={r.share_increased === 0 ? "ok" : "bad"}>{fmtPct(r.share_increased)}</Badge></td>
                <td className="num">{fmtPct(r.recall_full)} → {fmtPct(r.recall_missing)}</td></tr>))}
            </tbody>
          </table>
        </Panel>
        {nr && (
          <Panel title="Noisier detectors" description="800 new simulated sessions scored twice: as trained, and with detector false positives increased by 50%.">
            <table className="table">
              <thead><tr><th /><th className="num">PR-AUC</th><th className="num">Recall</th><th className="num">FPR</th></tr></thead>
              <tbody>{Object.entries(nr).filter(([k]) => k !== "change").map(([k, x]: [string, Any]) => (
                <tr key={k}><td className="small">{k}</td><td className="num">{x.pr_auc.toFixed(3)}</td><td className="num">{fmtPct(x.recall)}</td><td className="num">{fmtPct(x.fpr)}</td></tr>))}
              </tbody>
            </table>
          </Panel>
        )}
        <Panel title="Cost of a false accusation" description="Threshold that minimises expected cost when a false accusation costs N times a missed violation (chosen on calibration, measured on validation).">
          <table className="table">
            <thead><tr><th className="num">Cost ratio</th><th className="num">Recall</th><th className="num">FPR</th><th className="num">Cost / 1000</th></tr></thead>
            <tbody>{e.cost.map((c: Any) => (
              <tr key={c.fp_to_fn_cost_ratio}><td className="num">{c.fp_to_fn_cost_ratio} : 1</td><td className="num">{fmtPct(c.val_recall)}</td>
                <td className="num">{fmtPct(c.val_fpr)}</td><td className="num">{c.val_expected_cost_per_1000.toFixed(1)}</td></tr>))}
            </tbody>
          </table>
        </Panel>
      </div>
      {drift && (
        <Panel title="Drift monitor" description={`Batches of ${drift.batches[0]?.sessions ?? 200} sessions against the validation reference (review rate ${fmtPct(drift.reference.review_rate)}). Alerts: PSI > ${drift.thresholds.psi}, review rate beyond ×${drift.thresholds.review_rate_ratio}, or FPR clearly above budget.`} flush>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Source</th><th className="num">Batch</th><th className="num">Max PSI</th><th className="num">Review rate</th><th>Alerts</th></tr></thead>
              <tbody>{drift.batches.map((b: Any, i: number) => (
                <tr key={i}><td className="small">{b.source}</td><td className="num">{b.batch}</td>
                  <td className="num">{b.max_psi.toFixed(2)}</td><td className="num">{fmtPct(b.review_rate)}</td>
                  <td className="small">{b.alerts.length === 0 ? <Badge tone="ok" plain>None</Badge> : b.alerts.map((a: string) => <div key={a}><Badge tone="warn" plain>Alert</Badge> {a}</div>)}</td></tr>))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </>
  );
}
