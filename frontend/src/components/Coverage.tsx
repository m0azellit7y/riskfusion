import type { Coverage } from "../lib/api";
import { LIGHTING, WEBCAMS, WEBCAM_LABEL, titleCase } from "../lib/format";

/** FR-2 progress: >= 60 eligible sessions across >= 3 lighting and >= 2 webcam conditions. */
export function CoverageSummary({ c }: { c: Coverage }) {
  const pct = Math.min(1, c.eligible_sessions / c.target_sessions);
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <span>
          <strong className="num">{c.eligible_sessions}</strong> of {c.target_sessions} sessions recorded
        </span>
        <span className="muted small">
          {c.lighting_conditions.length}/{c.target_lighting_conditions} lighting and {c.webcam_conditions.length}/
          {c.target_webcam_conditions} webcam conditions
        </span>
      </div>
      <div className="progress" aria-label="Corpus progress">
        <span style={{ width: `${pct * 100}%` }} />
      </div>
      <div className="coverage" style={{ marginTop: 16 }}>
        <div />
        {WEBCAMS.map((w) => (
          <div className="head" key={w}>
            {WEBCAM_LABEL[w]}
          </div>
        ))}
        {LIGHTING.map((l) => (
          <div key={l} style={{ display: "contents" }}>
            <div className="rowhead">{titleCase(l)} light</div>
            {WEBCAMS.map((w) => {
              const n = c.grid[l]?.[w] ?? 0;
              return (
                <div key={w} className={`cell ${n > 0 ? "has" : ""}`} title={`${n} sessions`}>
                  {n}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
        Counts only consented, uploaded, non-rehearsal sessions of at least 20 minutes.
      </p>
    </div>
  );
}
