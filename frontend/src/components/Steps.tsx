export const SESSION_STEPS = ["Set up", "Confirm consent", "Check equipment", "Record", "Review and upload"];

export function Steps({ current, steps = SESSION_STEPS }: { current: number; steps?: string[] }) {
  return (
    <ol className="steps" aria-label="Progress">
      {steps.map((s, i) => (
        <li key={s} className={`step ${i === current ? "current" : i < current ? "done" : ""}`} aria-current={i === current ? "step" : undefined}>
          <span className="step-dot">{i < current ? "✓" : i + 1}</span>
          {s}
        </li>
      ))}
    </ol>
  );
}
