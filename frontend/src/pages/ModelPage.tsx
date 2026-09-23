import { IntervalRows, LineChart } from "../components/Charts";
import { Badge, ErrorNotice, Facts, Loading, PageHeader, Panel, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { featureLabel, fmtDateTime, fmtPct, titleCase } from "../lib/format";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

export default function ModelPage() {
  const info = useAsync(() => api.get<Any>("/model-info"), []);
  const ev = useAsync(() => api.get<Any>("/model/evaluation"), []);
  if (info.loading || ev.loading) return <Loading />;
  if (info.error || !info.data) return <ErrorNotice message={info.error ?? "No model"} onRetry={info.reload} />;
  const m = info.data;
  const e = ev.data?.evaluation;
  const tr = ev.data?.training;
  const h1 = (ev.data?.holdout ?? []).filter(Boolean);
  const v = m.headline_metrics_validation;
  const families = [
    { label: "Rule baseline", v: tr?.baseline?.validation },
    { label: "Temporal (ROCKET)", v: tr?.temporal?.validation },
    { label: "Fusion (LightGBM)", v: tr?.primary?.validation },
  ];
  return (
    <>
      <PageHeader
        title="Model"
        description={`${m.model_version}, trained ${fmtDateTime(m.trained_at)} on ${m.dataset_version}. Selected on the validation split; calibrated on the calibration split.`}
      />
      <Facts
        items={[
          { label: "PR-AUC (validation)", value: v.pr_auc.toFixed(3), note: `baseline ${tr?.baseline?.validation?.pr_auc?.toFixed(3)}` },
          { label: "Recall at operating point", value: fmtPct(v.recall), note: "target ≥ 80%" },
          { label: "False-positive rate", value: fmtPct(v.fpr), note: `budget ${fmtPct(m.operating_point.fpr_budget, 0)} (reviewer capacity)` },
          { label: "Calibration error (ECE)", value: v.ece.toFixed(3), note: `target < 0.05, ${m.calibration}` },
          { label: "Sealed holdout", value: `${h1.length} of 2`, note: "accesses used" },
        ]}
      />
      <div className="grid-2" style={{ marginTop: 20 }}>
        <Panel title="Calibration" description="Predicted risk against the observed violation rate (validation, 15 bins). On the diagonal = well calibrated.">
          {e && (
            <LineChart
              diagonal
              xDomain={[0, 1]}
              xLabel="Predicted risk"
              yLabel="Observed rate"
              series={[{ name: "model", color: "#17324d", dots: true,
                points: e.reliability.map((b: Any) => [b.mean_pred, b.frac_pos] as [number, number]) }]}
            />
          )}
        </Panel>
        <Panel title="Review tiers" description="Thresholds on calibrated risk, set from the share of clean sessions each tier may send to reviewers.">
          <table className="table">
            <thead><tr><th>Tier</th><th className="num">Risk at or above</th><th className="num">Clean sessions sent</th></tr></thead>
            <tbody>
              {(["PRIORITY_REVIEW", "HUMAN_REVIEW", "ROUTINE_REVIEW"] as const).map((t) => (
                <tr key={t}>
                  <td>{titleCase(t)}</td>
                  <td className="num">{m.tiers[t].toFixed(3)}</td>
                  <td className="num">{t === "PRIORITY_REVIEW" ? "≤ 0.5%" : t === "HUMAN_REVIEW" ? "≤ 3%" : "≤ 10%"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small muted" style={{ marginBottom: 0, marginTop: 10 }}>
            Scores below the routine threshold get no action. The output is always a recommendation for a person.
          </p>
        </Panel>
        <Panel title="Model comparison" description="Validation split. Recall at each model's own 3% operating point.">
          <table className="table">
            <thead><tr><th>Model</th><th className="num">PR-AUC</th><th className="num">Recall</th><th className="num">ECE</th></tr></thead>
            <tbody>
              {families.map((f) => f.v && (
                <tr key={f.label}><td>{f.label}</td><td className="num">{f.v.pr_auc.toFixed(3)}</td>
                  <td className="num">{fmtPct(f.v.recall)}</td><td className="num">{f.v.ece.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="Sealed holdout" description="Evaluated at most twice (end of Phase 4 and final). Every access is logged.">
          {h1.length === 0 ? <p className="muted">Not opened.</p> : (
            <table className="table">
              <thead><tr><th>Access</th><th className="num">PR-AUC (95% CI)</th><th className="num">Recall</th><th className="num">FPR</th></tr></thead>
              <tbody>
                {h1.map((h: Any) => (
                  <tr key={h.access}>
                    <td className="small">{h.access}: {h.purpose}</td>
                    <td className="num">{h.primary.pr_auc.toFixed(3)} <span className="faint">({h.primary.pr_auc_ci.map((x: number) => x.toFixed(2)).join("–")})</span></td>
                    <td className="num">{fmtPct(h.primary.recall)}</td>
                    <td className="num">{fmtPct(h.primary.fpr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
      {e && (
        <>
          <Panel title="What drives the score" description="Mean absolute SHAP contribution on the validation split (top 15).">
            <IntervalRows
              max={Math.max(...e.interpretation.importance.slice(0, 15).map((i: Any) => i.mean_abs_shap))}
              format={(x) => x.toFixed(3)}
              rows={e.interpretation.importance.slice(0, 15).map((i: Any) => ({ label: featureLabel(i.feature).slice(0, 22), value: i.mean_abs_shap }))}
            />
          </Panel>
          <Panel title="Partial dependence" description="Average calibrated risk as one feature varies, others held as observed (top features).">
            <div className="grid-3">
              {e.interpretation.partial_dependence.slice(0, 6).map((p: Any) => (
                <div key={p.feature}>
                  <h3 className="small" style={{ marginBottom: 4 }}>{featureLabel(p.feature)}</h3>
                  <LineChart height={150} xLabel="feature value" yLabel="risk" yDomain={[0, Math.max(0.1, ...p.mean_risk) * 1.1]}
                    series={[{ name: p.feature, color: "#2c4a67", points: p.grid.map((g: number, i: number) => [g, p.mean_risk[i]] as [number, number]) }]} />
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Ablations" description="Each variant retrained with the same settings and scored on validation." flush>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Group</th><th>Variant</th><th className="num">Features</th><th className="num">PR-AUC</th><th className="num">Change</th><th className="num">Recall</th></tr></thead>
                <tbody>
                  {e.ablations.map((a: Any) => {
                    const ref = e.ablations[0].val_pr_auc;
                    const d = a.val_pr_auc - ref;
                    return (
                      <tr key={a.variant}>
                        <td className="small muted">{a.group}</td><td>{a.variant}</td><td className="num">{a.n_features}</td>
                        <td className="num">{a.val_pr_auc.toFixed(3)}</td>
                        <td className="num">{a.group === "reference" ? "—" : <Badge plain tone={d < -0.03 ? "warn" : "neutral"}>{d >= 0 ? "+" : ""}{d.toFixed(3)}</Badge>}</td>
                        <td className="num">{fmtPct(a.val_recall_at_op)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Flags" description="Time-bounded flags against the true episodes (validation). Target: IoU ≥ 0.5 for at least 70% of episodes.">
            <IntervalRows max={1} reference={{ value: 0.7, label: "target" }}
              rows={Object.entries(e.flags.by_type).map(([k, v]: [string, Any]) => ({ label: titleCase(k), value: v.share_iou_ge_0_5 }))} />
            <p className="small muted" style={{ margin: 0 }}>
              Overall {fmtPct(e.flags.share_iou_ge_0_5)} of {e.flags.episodes} episodes. Flags are shown to reviewers only for sessions recommended for review.
            </p>
          </Panel>
          <Panel title="Experiment log" flush>
            <table className="table"><tbody>
              {(ev.data?.runs ?? []).map((r: Any) => (
                <tr key={r.run_id}><td className="small">{r.run_id}</td>
                  <td className="small muted">{Object.entries(r.metrics).filter(([k]) => k.includes("pr_auc")).map(([k, x]) => `${k} ${(x as number).toFixed(3)}`).join(", ")}</td></tr>
              ))}
            </tbody></table>
          </Panel>
        </>
      )}
    </>
  );
}
