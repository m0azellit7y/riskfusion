// Small, dependency-free SVG charts sized to their container.
import { useEffect, useRef, useState } from "react";

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(480);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  diagonal,
  height = 220,
  xDomain,
  yDomain = [0, 1],
}: {
  series: { name: string; color: string; points: [number, number][]; dots?: boolean }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  height?: number;
  xDomain?: [number, number];
  yDomain?: [number, number];
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const pad = { l: 44, r: 12, t: 10, b: 34 };
  const all = series.flatMap((s) => s.points);
  const [x0, x1] = xDomain ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
  const [y0, y1] = yDomain;
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => height - pad.b - ((y - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div ref={ref}>
      <svg width={W} height={height} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(y0 + t * (y1 - y0))} y2={sy(y0 + t * (y1 - y0))} stroke="#eef1f4" />
            <text x={pad.l - 6} y={sy(y0 + t * (y1 - y0)) + 4} fontSize="11" fill="#8a96a3" textAnchor="end">
              {(y0 + t * (y1 - y0)).toFixed(2)}
            </text>
            <text x={sx(x0 + t * (x1 - x0))} y={height - pad.b + 16} fontSize="11" fill="#8a96a3" textAnchor="middle">
              {(x0 + t * (x1 - x0)).toFixed(x1 - x0 > 10 ? 0 : 2)}
            </text>
          </g>
        ))}
        {diagonal && <line x1={sx(x0)} y1={sy(y0)} x2={sx(x1)} y2={sy(y1)} stroke="#c3cbd4" strokeDasharray="4 4" />}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} points={s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />
            {s.dots && s.points.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3} fill={s.color} />)}
          </g>
        ))}
        <text x={(W + pad.l) / 2} y={height - 2} fontSize="12" fill="#5a6878" textAnchor="middle">{xLabel}</text>
        <text x={12} y={height / 2} fontSize="12" fill="#5a6878" textAnchor="middle" transform={`rotate(-90 12 ${height / 2})`}>
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

/** Horizontal interval chart: a point estimate with a confidence interval per row (e.g. slice FPR). */
export function IntervalRows({
  rows,
  max,
  reference,
  format = (v: number) => `${(v * 100).toFixed(1)}%`,
}: {
  rows: { label: string; value: number; lo?: number; hi?: number; note?: string }[];
  max: number;
  reference?: { value: number; label: string };
  format?: (v: number) => string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const left = 130;
  const right = 70;
  const rowH = 26;
  const H = rows.length * rowH + 20;
  const sx = (v: number) => left + (Math.min(v, max) / max) * (W - left - right);
  return (
    <div ref={ref}>
      <svg width={W} height={H} role="img" aria-label="Interval chart">
        {reference && (
          <g>
            <line x1={sx(reference.value)} x2={sx(reference.value)} y1={0} y2={H - 16} stroke="#9a5f0e" strokeDasharray="4 3" />
            <text x={sx(reference.value)} y={H - 4} fontSize="11" fill="#9a5f0e" textAnchor="middle">{reference.label}</text>
          </g>
        )}
        {rows.map((r, i) => {
          const y = i * rowH + 14;
          return (
            <g key={r.label}>
              <text x={0} y={y + 4} fontSize="12.5" fill="#1f2933">{r.label}</text>
              {r.lo != null && r.hi != null && (
                <line x1={sx(r.lo)} x2={sx(r.hi)} y1={y} y2={y} stroke="#9fb3c8" strokeWidth={6} strokeLinecap="round" />
              )}
              <circle cx={sx(r.value)} cy={y} r={5} fill="#17324d" />
              <text x={W - right + 8} y={y + 4} fontSize="12" fill="#5a6878">{format(r.value)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
