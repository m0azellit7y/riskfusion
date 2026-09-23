import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { SessionStatus } from "../lib/api";
import { STATUS_LABEL, statusTone } from "../lib/format";

// ---------- data loading ----------
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    fn()
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload, setData };
}

// ---------- layout pieces ----------
export function PageHeader(props: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  crumbs?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div>
        {props.crumbs && <div className="crumbs">{props.crumbs}</div>}
        <h1>{props.title}</h1>
        {props.description && <p>{props.description}</p>}
      </div>
      {props.actions && <div className="actions">{props.actions}</div>}
    </header>
  );
}

export function Panel(props: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
}) {
  return (
    <section className={`panel ${props.className ?? ""}`}>
      {(props.title || props.actions) && (
        <div className="panel-head">
          <div>
            {props.title && <h2>{props.title}</h2>}
            {props.description && <p>{props.description}</p>}
          </div>
          {props.actions && <div className="row">{props.actions}</div>}
        </div>
      )}
      <div className={`panel-body ${props.flush ? "flush" : ""}`}>{props.children}</div>
    </section>
  );
}

export function Facts({ items }: { items: { label: string; value: ReactNode; note?: ReactNode }[] }) {
  return (
    <div className="facts">
      {items.map((f) => (
        <div className="fact" key={f.label}>
          <div className="fact-label">{f.label}</div>
          <div className="fact-value">{f.value}</div>
          {f.note && <div className="fact-note">{f.note}</div>}
        </div>
      ))}
    </div>
  );
}

export function Badge({ tone = "neutral", children, plain }: { tone?: string; children: ReactNode; plain?: boolean }) {
  return <span className={`badge badge-${tone} ${plain ? "plain" : ""}`}>{children}</span>;
}

export function StatusBadge({ status }: { status: SessionStatus }) {
  return <Badge tone={statusTone(status)}>{STATUS_LABEL[status] ?? status}</Badge>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" aria-hidden />
      {label}
    </div>
  );
}

export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="notice notice-bad" role="alert">
      <strong>This could not be loaded.</strong>
      {message}
      {onRetry && (
        <div style={{ marginTop: 8 }}>
          <button className="btn btn-sm" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Field(props: { label: string; hint?: ReactNode; error?: string | null; htmlFor?: string; children: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={props.htmlFor}>{props.label}</label>
      {props.children}
      {props.hint && <div className="hint">{props.hint}</div>}
      {props.error && <div className="error">{props.error}</div>}
    </div>
  );
}

// ---------- modal ----------
export function Modal(props: {
  title: string;
  description?: ReactNode;
  onClose: () => void;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const { onClose } = props;
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const first = ref.current?.querySelector<HTMLElement>("input, select, textarea, button");
    first?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      prev?.focus();
    };
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${props.wide ? "wide" : ""}`} role="dialog" aria-modal="true" aria-label={props.title} ref={ref}>
        <div className="modal-head">
          <h2>{props.title}</h2>
          {props.description && <p>{props.description}</p>}
        </div>
        {props.children && <div className="modal-body">{props.children}</div>}
        {props.footer && <div className="modal-foot">{props.footer}</div>}
      </div>
    </div>
  );
}

// ---------- toasts ----------
type Toast = { id: number; text: string; bad?: boolean };
const ToastCtx = createContext<(text: string, bad?: boolean) => void>(() => {});
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, bad = false) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, bad }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), bad ? 7000 : 4000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.bad ? "bad" : ""}`}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}
export const useToast = () => useContext(ToastCtx);

// ---------- minimal markdown (consent form) ----------
export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let list: { ordered: boolean; items: string[] } | null = null;
  const inline = (s: string, key: number) => {
    const parts = s.split(/(\*\*[^*]+\*\*)/g);
    return (
      <span key={key}>
        {parts.map((p, i) => (p.startsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong> : p))}
      </span>
    );
  };
  const flush = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={blocks.length}>
        {list.items.map((it, i) => (
          <li key={i}>{inline(it, i)}</li>
        ))}
      </Tag>,
    );
    list = null;
  };
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) blocks.push(<p key={blocks.length}>{inline(para.join(" "), 0)}</p>);
    para = [];
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const ul = /^- (.*)/.exec(line);
    const ol = /^\d+\. (.*)/.exec(line);
    if (line.startsWith("### ")) {
      flushPara();
      flush();
      blocks.push(<h3 key={blocks.length}>{line.slice(4)}</h3>);
    } else if (line.startsWith("## ")) {
      flushPara();
      flush();
      blocks.push(<h2 key={blocks.length}>{line.slice(3)}</h2>);
    } else if (ul || ol) {
      flushPara();
      const ordered = !!ol;
      if (!list || list.ordered !== ordered) {
        flush();
        list = { ordered, items: [] };
      }
      list.items.push((ul ?? ol)![1]);
    } else if (line === "") {
      flushPara();
      flush();
    } else if (list && raw.startsWith("  ")) {
      list.items[list.items.length - 1] += " " + line.trim();
    } else {
      flush();
      para.push(line);
    }
  }
  flushPara();
  flush();
  return <>{blocks}</>;
}

// ---------- icons (inline, 18px) ----------
const P: Record<string, string> = {
  overview: "M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z",
  sessions: "M4 6h16M4 12h16M4 18h10",
  record: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z",
  people: "M16 11a4 4 0 1 0-8 0M4 21a8 8 0 0 1 16 0",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  system: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.8 1.2V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-2.8-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0-1.2-2.8H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.2-2.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 2.8-1.2V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 2.8 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0 1.2 2.8H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
  menu: "M3 6h18M3 12h18M3 18h18",
  plus: "M12 5v14M5 12h14",
};
export function Icon({ name, size = 18 }: { name: keyof typeof P | string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={P[name] ?? ""} />
    </svg>
  );
}
