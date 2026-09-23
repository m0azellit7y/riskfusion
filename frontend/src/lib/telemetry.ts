// Browser telemetry capture during a mock session (SRS FR-5), emitted as event.v1.
// Privacy: pasted text and keys are never recorded — only paste length and keystroke counts.

import { api } from "./api";

const DETECTOR = "browser_telemetry";
const VERSION = "web-1.0";
const FLUSH_MS = 5000;
const INPUT_WINDOW_MS = 10000;

type Channel = "screen" | "device" | "behavioral";
interface OutEvent {
  event_uid: string;
  schema: "event.v1";
  session_id: string;
  ts_ms: number;
  channel: Channel;
  detector: string;
  detector_version: string;
  event_type: string;
  payload: Record<string, unknown>;
  confidence: number;
}
export interface TelemetryStats {
  captured: number;
  sent: number;
  pending: number;
  rejected: number;
  lastError: string | null;
  monitorCount: number | null;
}

function uid(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID().replace(/-/g, "")
    : Math.random().toString(16).slice(2) + Date.now().toString(16);
}

export class TelemetryCollector {
  private queue: OutEvent[] = [];
  private keys = 0;
  private windowStart = 0;
  private timers: number[] = [];
  private flushing = false;
  private cleanup: (() => void)[] = [];
  stats: TelemetryStats = { captured: 0, sent: 0, pending: 0, rejected: 0, lastError: null, monitorCount: null };
  onChange?: (s: TelemetryStats) => void;

  constructor(private sessionId: string, private t0: number) {}

  private now(): number {
    return Math.max(0, Math.round(performance.now() - this.t0));
  }

  private emit(channel: Channel, event_type: string, payload: Record<string, unknown>, ts = this.now()) {
    this.queue.push({
      event_uid: uid(),
      schema: "event.v1",
      session_id: this.sessionId,
      ts_ms: ts,
      channel,
      detector: DETECTOR,
      detector_version: VERSION,
      event_type,
      payload,
      confidence: 1,
    });
    this.stats.captured += 1;
    this.stats.pending = this.queue.length;
    this.onChange?.({ ...this.stats });
  }

  start() {
    this.windowStart = this.now();
    this.emit("screen", "TAB_VISIBILITY", { hidden: document.visibilityState === "hidden" });
    this.emit("screen", "FULLSCREEN_CHANGE", { active: document.fullscreenElement != null });
    const ext = (window.screen as Screen & { isExtended?: boolean }).isExtended;
    if (typeof ext === "boolean") {
      this.stats.monitorCount = ext ? 2 : 1;
      this.emit("device", "MONITOR_COUNT", { count: ext ? 2 : 1 });
    } else {
      // Firefox/Safari do not expose multi-monitor information (see SRS_AUDIT A-13).
      this.emit("device", "DEVICE_UNKNOWN", { reason: "not_observable" });
    }
    const onVis = () => this.emit("screen", "TAB_VISIBILITY", { hidden: document.visibilityState === "hidden" });
    const onFs = () => this.emit("screen", "FULLSCREEN_CHANGE", { active: document.fullscreenElement != null });
    const onPaste = (e: ClipboardEvent) => {
      const len = e.clipboardData?.getData("text")?.length ?? 0;
      this.emit("screen", "PASTE", { length: len });
    };
    const onKey = () => {
      this.keys += 1;
    };
    document.addEventListener("visibilitychange", onVis);
    document.addEventListener("fullscreenchange", onFs);
    document.addEventListener("paste", onPaste, true);
    document.addEventListener("keydown", onKey, true);
    const scr = window.screen as Screen & EventTarget;
    const onScreen = () => {
      const e2 = (window.screen as Screen & { isExtended?: boolean }).isExtended;
      if (typeof e2 === "boolean") this.emit("device", "MONITOR_COUNT", { count: e2 ? 2 : 1 });
    };
    if (typeof scr.addEventListener === "function") scr.addEventListener("change", onScreen);
    this.cleanup.push(() => {
      document.removeEventListener("visibilitychange", onVis);
      document.removeEventListener("fullscreenchange", onFs);
      document.removeEventListener("paste", onPaste, true);
      document.removeEventListener("keydown", onKey, true);
      if (typeof scr.removeEventListener === "function") scr.removeEventListener("change", onScreen);
    });
    this.timers.push(window.setInterval(() => this.closeInputWindow(), INPUT_WINDOW_MS));
    this.timers.push(window.setInterval(() => void this.flush(), FLUSH_MS));
  }

  private closeInputWindow(final = false) {
    const now = this.now();
    const windowS = Math.max(0.001, (now - this.windowStart) / 1000);
    if (!final || windowS >= 1) {
      this.emit("behavioral", "INPUT_ACTIVITY", { keystrokes: this.keys, window_s: Math.round(windowS * 1000) / 1000 });
    }
    this.keys = 0;
    this.windowStart = now;
  }

  async flush(): Promise<void> {
    if (this.flushing || this.queue.length === 0) return;
    this.flushing = true;
    const batch = this.queue.slice(0, 500);
    try {
      const res = await api.post<{ accepted: number; duplicates: number; rejected: number }>(
        `/sessions/${this.sessionId}/events`,
        { events: batch },
      );
      this.queue.splice(0, batch.length);
      this.stats.sent += res.accepted + res.duplicates;
      this.stats.rejected += res.rejected;
      this.stats.lastError = null;
    } catch (e) {
      // keep the batch; it is retried on the next flush (event_uid makes re-delivery safe)
      this.stats.lastError = e instanceof Error ? e.message : "Telemetry upload failed";
    } finally {
      this.stats.pending = this.queue.length;
      this.flushing = false;
      this.onChange?.({ ...this.stats });
    }
  }

  async stop(): Promise<void> {
    this.timers.forEach((t) => window.clearInterval(t));
    this.timers = [];
    this.closeInputWindow(true);
    this.cleanup.forEach((f) => f());
    this.cleanup = [];
    for (let i = 0; i < 5 && this.queue.length > 0; i++) {
      await this.flush();
      if (this.queue.length > 0) await new Promise((r) => setTimeout(r, 800));
    }
  }
}
