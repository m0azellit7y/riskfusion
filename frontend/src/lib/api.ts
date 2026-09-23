// Typed client for the RiskFusion API. All requests go to /api (proxied to the backend).

export const API = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function operatorName(): string {
  try {
    return localStorage.getItem("rf.operator") || "operator";
  } catch {
    return "operator";
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API + path, {
      method,
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        "X-Operator": operatorName(),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(0, "The server could not be reached. Check that the API is running.");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    const msg = typeof detail === "string" ? detail : `The request failed (${res.status}).`;
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  get: <T,>(p: string) => request<T>("GET", p),
  post: <T,>(p: string, b: unknown = {}) => request<T>("POST", p, b),
  put: <T,>(p: string, b: unknown) => request<T>("PUT", p, b),
};

/** Multipart upload with progress (fetch cannot report upload progress). */
export function uploadFile(
  path: string,
  fields: Record<string, string>,
  file: Blob,
  filename: string,
  onProgress?: (fraction: number) => void,
): { promise: Promise<unknown>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<unknown>((resolve, reject) => {
    const form = new FormData();
    Object.entries(fields).forEach(([k, v]) => form.append(k, v));
    form.append("file", file, filename);
    xhr.open("POST", API + path);
    xhr.setRequestHeader("X-Operator", operatorName());
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let data: { detail?: string } | null = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        data = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(xhr.status, data?.detail || `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new ApiError(0, "The upload was interrupted. Check the connection and try again."));
    xhr.onabort = () => reject(new ApiError(0, "Upload cancelled."));
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

// ---- types (mirror backend/riskfusion_api/schemas.py) ----
export type SessionStatus =
  | "GENERATED" | "CREATED" | "CONSENTED" | "RECORDING" | "RECORDED" | "UPLOADED" | "PROCESSING"
  | "ANALYZING" | "COMPLETED" | "FAILED" | "REVIEWED" | "DELETED";

export interface Participant {
  id: string;
  code: string;
  status: "ACTIVE" | "WITHDRAWN";
  adult_confirmed: boolean;
  created_at: string;
  withdrawn_at: string | null;
  notes: string | null;
  consent_active: boolean;
  consent_version: string | null;
  session_count: number;
}
export interface Consent {
  id: string;
  consent_version: string;
  signed_name: string;
  ethics_reference: string | null;
  witnessed_by: string | null;
  signed_at: string;
  withdrawn_at: string | null;
  withdrawal_reason: string | null;
}
export interface ParticipantDetail extends Participant {
  consents: Consent[];
  has_demographics: boolean;
}
export interface Session {
  id: string;
  source: "SIMULATED" | "MOCK";
  status: SessionStatus;
  participant_id: string | null;
  participant_code: string | null;
  helper_participant_id: string | null;
  dataset_version: string | null;
  script_id: string | null;
  script_version: string | null;
  is_rehearsal: boolean;
  behavior_profile: string | null;
  lighting: string | null;
  webcam_class: string | null;
  room_noise: string | null;
  connection_stability: string | null;
  eyewear: boolean | null;
  head_covering: boolean | null;
  duration_s: number | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  ended_at: string | null;
  violation_label: boolean | null;
  split: string | null;
}
export interface Recording {
  id: string;
  kind: "webcam_av" | "enrollment_image";
  filename: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  status: "STORED" | "PURGED";
  created_at: string;
  retention_until: string | null;
  purged_at: string | null;
}
export interface Episode {
  episode_no: number;
  violation_type: string;
  instruction: string;
  scheduled_start_ms: number;
  scheduled_end_ms: number;
  shown_at_ms: number | null;
  completed_at_ms: number | null;
}
export interface Interval {
  start_ms: number;
  end_ms: number;
  type: string;
}
export interface Label {
  source: string;
  violation: boolean;
  violation_types: string[] | null;
  intervals: Interval[] | null;
  labeler: string | null;
  labeled_at: string | null;
  confidence: string | null;
}
export interface SessionDetail extends Session {
  notes: string | null;
  browser: Record<string, unknown> | null;
  channels_outage: string[] | null;
  baseline_movement: number | null;
  history: { from_status: string | null; to_status: string; actor: string; note: string | null; at: string }[];
  recordings: Recording[];
  episodes: Episode[];
  label: Label | null;
  event_counts: Record<string, number>;
  dead_letter_count: number;
}
export interface Page<T> {
  total: number;
  items: T[];
}
export interface ScriptDef {
  title: string;
  violation: boolean;
  profile: string;
  summary: string;
  requires_helper?: boolean;
  rehearsal?: boolean;
  episodes: { type: string; start_s: number; duration_s: number; instruction: string }[];
}
export interface ScriptsDoc {
  script_version: string;
  min_duration_s: number;
  max_duration_s: number;
  scripts: Record<string, ScriptDef>;
}
export interface Coverage {
  eligible_sessions: number;
  target_sessions: number;
  lighting_conditions: string[];
  webcam_conditions: string[];
  target_lighting_conditions: number;
  target_webcam_conditions: number;
  grid: Record<string, Record<string, number>>;
  by_script: Record<string, number>;
  active_participants: number;
  participants_recorded: number;
  met: boolean;
}
export interface TelemetryEvent {
  schema: string;
  session_id: string;
  ts_ms: number;
  channel: string;
  detector: string;
  detector_version: string;
  event_type: string;
  payload: Record<string, unknown>;
  confidence: number;
  quality: Record<string, unknown> | null;
}
