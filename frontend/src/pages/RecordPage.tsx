import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Steps } from "../components/Steps";
import { Timeline } from "../components/Timeline";
import { Badge, ErrorNotice, Loading, Modal, PageHeader, Panel, useAsync, useToast } from "../components/ui";
import { api, uploadFile, type Episode, type ScriptsDoc, type SessionDetail } from "../lib/api";
import { fmtBytes, fmtClock, VIOLATION_LABEL } from "../lib/format";
import { QUIZ } from "../lib/quiz";
import { TelemetryCollector, type TelemetryStats } from "../lib/telemetry";

type Phase = "check" | "recording" | "review";
const MIME_CANDIDATES = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
const CUE_SHOW_MS_WHEN_OPEN_ENDED = 20000;

function pickMime(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) ?? null;
}

export default function RecordPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const sess = useAsync(() => api.get<SessionDetail>(`/sessions/${id}`), [id]);
  const scripts = useAsync(() => api.get<ScriptsDoc>("/mock-scripts"), []);

  const [phase, setPhase] = useState<Phase>("check");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [devices, setDevices] = useState<{ cams: MediaDeviceInfo[]; mics: MediaDeviceInfo[] }>({ cams: [], mics: [] });
  const [camId, setCamId] = useState("");
  const [micId, setMicId] = useState("");
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [micHeard, setMicHeard] = useState(false);
  const [resolution, setResolution] = useState<{ w: number; h: number } | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [photoBusy, setPhotoBusy] = useState(false);
  const [useFullscreen, setUseFullscreen] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [trackLost, setTrackLost] = useState<string | null>(null);
  const [tele, setTele] = useState<TelemetryStats | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [durationS, setDurationS] = useState(0);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [confirmStop, setConfirmStop] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const t0Ref = useRef(0);
  const teleRef = useRef<TelemetryCollector | null>(null);
  const audioRef = useRef<{ ctx: AudioContext; raf: number } | null>(null);
  const mimeRef = useRef<string | null>(pickMime());
  const streamRef = useRef<MediaStream | null>(null);
  const shownRef = useRef<Set<number>>(new Set());
  const doneRef = useRef<Set<number>>(new Set());

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStream(null);
    if (audioRef.current) {
      cancelAnimationFrame(audioRef.current.raf);
      void audioRef.current.ctx.close();
      audioRef.current = null;
    }
  }, []);

  // --- camera & microphone ------------------------------------------------------------------
  const openDevices = useCallback(
    async (cam?: string, mic?: string) => {
      setDeviceError(null);
      stopStream();
      if (!navigator.mediaDevices?.getUserMedia) {
        setDeviceError("This browser cannot access a camera. Use a current version of Chrome, Edge or Firefox over http://localhost or https.");
        return;
      }
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: cam ? { exact: cam } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: { deviceId: mic ? { exact: mic } : undefined },
        });
        streamRef.current = s;
        setStream(s);
        const vt = s.getVideoTracks()[0];
        const st = vt?.getSettings();
        if (st?.width && st?.height) setResolution({ w: st.width, h: st.height });
        const all = await navigator.mediaDevices.enumerateDevices();
        setDevices({ cams: all.filter((d) => d.kind === "videoinput"), mics: all.filter((d) => d.kind === "audioinput") });
        setCamId(vt?.getSettings().deviceId ?? "");
        setMicId(s.getAudioTracks()[0]?.getSettings().deviceId ?? "");
        // level meter
        const ctx = new AudioContext();
        const src = ctx.createMediaStreamSource(s);
        const an = ctx.createAnalyser();
        an.fftSize = 1024;
        src.connect(an);
        const buf = new Float32Array(an.fftSize);
        const tick = () => {
          an.getFloatTimeDomainData(buf);
          let sum = 0;
          for (const v of buf) sum += v * v;
          const rms = Math.sqrt(sum / buf.length);
          const lvl = Math.min(1, rms * 6);
          setLevel(lvl);
          if (lvl > 0.12) setMicHeard(true);
          if (audioRef.current) audioRef.current.raf = requestAnimationFrame(tick);
        };
        audioRef.current = { ctx, raf: requestAnimationFrame(tick) };
        s.getTracks().forEach((t) =>
          t.addEventListener("ended", () => setTrackLost(t.kind === "video" ? "The camera stopped working." : "The microphone stopped working.")),
        );
      } catch (e) {
        const name = (e as DOMException).name;
        setDeviceError(
          name === "NotAllowedError"
            ? "Camera or microphone access was blocked. Allow access in the browser's address bar, then try again."
            : name === "NotFoundError"
              ? "No camera or microphone was found. Connect one and try again."
              : name === "NotReadableError"
                ? "The camera is in use by another application. Close it and try again."
                : `The camera could not be opened (${name}).`,
        );
      }
    },
    [stopStream],
  );

  useEffect(() => {
    if (videoRef.current && stream && phase !== "review") {
      videoRef.current.srcObject = stream;
      void videoRef.current.play().catch(() => undefined);
    }
  }, [stream, phase]);

  useEffect(() => () => stopStream(), [stopStream]);

  // warn before leaving while recording or with an unsent recording
  useEffect(() => {
    if (phase === "check") return;
    const h = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [phase]);

  // --- enrolment photo ------------------------------------------------------------------------
  async function takePhoto() {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    setPhotoBusy(true);
    try {
      const c = document.createElement("canvas");
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext("2d")!.drawImage(v, 0, 0);
      const b = await new Promise<Blob>((res, rej) => c.toBlob((x) => (x ? res(x) : rej(new Error("capture failed"))), "image/jpeg", 0.9));
      await uploadFile(`/sessions/${id}/recordings`, { kind: "enrollment_image", width: String(c.width), height: String(c.height) }, b, "enrolment.jpg").promise;
      if (photoUrl) URL.revokeObjectURL(photoUrl);
      setPhotoUrl(URL.createObjectURL(b));
      toast("Enrolment photo saved.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setPhotoBusy(false);
    }
  }

  // --- recording ------------------------------------------------------------------------------
  async function start() {
    const s = streamRef.current;
    const mime = mimeRef.current;
    if (!s || !mime) return;
    setBusy(true);
    try {
      if (useFullscreen && document.documentElement.requestFullscreen) {
        await document.documentElement.requestFullscreen().catch(() => undefined);
      }
      const ext = (window.screen as Screen & { isExtended?: boolean }).isExtended;
      const started = await api.post<SessionDetail>(`/sessions/${id}/start`, {
        browser: { userAgent: navigator.userAgent, mimeType: mime, resolution, screenIsExtended: ext ?? null, language: navigator.language },
      });
      setEpisodes(started.episodes);
      shownRef.current = new Set();
      doneRef.current = new Set();
      chunksRef.current = [];
      const rec = new MediaRecorder(s, { mimeType: mime, videoBitsPerSecond: 1_200_000, audioBitsPerSecond: 64_000 });
      rec.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      recorderRef.current = rec;
      t0Ref.current = performance.now();
      rec.start(1000);
      const tc = new TelemetryCollector(id, t0Ref.current);
      tc.onChange = setTele;
      tc.start();
      teleRef.current = tc;
      setPhase("recording");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  // clock + script cues
  useEffect(() => {
    if (phase !== "recording") return;
    const iv = window.setInterval(() => {
      const ms = performance.now() - t0Ref.current;
      setElapsed(ms);
      for (const ep of episodes) {
        if (ms >= ep.scheduled_start_ms && !shownRef.current.has(ep.episode_no)) {
          shownRef.current.add(ep.episode_no);
          void api.post(`/sessions/${id}/episodes/${ep.episode_no}`, { shown_at_ms: Math.round(ms) }).catch(() => undefined);
        }
        if (ep.scheduled_end_ms > 0 && ms >= ep.scheduled_end_ms && shownRef.current.has(ep.episode_no) && !doneRef.current.has(ep.episode_no)) {
          doneRef.current.add(ep.episode_no);
          void api.post(`/sessions/${id}/episodes/${ep.episode_no}`, { completed_at_ms: Math.round(ms) }).catch(() => undefined);
        }
      }
    }, 250);
    return () => window.clearInterval(iv);
  }, [phase, episodes, id]);

  async function stop() {
    const rec = recorderRef.current;
    if (!rec) return;
    setConfirmStop(false);
    setBusy(true);
    const dur = (performance.now() - t0Ref.current) / 1000;
    const done = new Promise<void>((res) => (rec.onstop = () => res()));
    rec.stop();
    await done;
    const b = new Blob(chunksRef.current, { type: mimeRef.current ?? "video/webm" });
    try {
      await api.post(`/sessions/${id}/stop`, { duration_s: Math.round(dur * 10) / 10 });
    } catch (e) {
      toast((e as Error).message, true);
    }
    await teleRef.current?.stop();
    stopStream();
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => undefined);
    setDurationS(dur);
    setBlob(b);
    setBlobUrl(URL.createObjectURL(b));
    setPhase("review");
    setBusy(false);
  }

  async function upload() {
    if (!blob) return;
    setUploadError(null);
    setUploadPct(0);
    try {
      await uploadFile(
        `/sessions/${id}/recordings`,
        { kind: "webcam_av", duration_s: String(Math.round(durationS * 10) / 10), ...(resolution ? { width: String(resolution.w), height: String(resolution.h) } : {}) },
        blob,
        `${id}.webm`,
        setUploadPct,
      ).promise;
      toast("Recording uploaded and ground-truth label created.");
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      nav(`/sessions/${id}`);
    } catch (e) {
      setUploadError((e as Error).message);
      setUploadPct(null);
    }
  }

  async function discard(reason: string) {
    setBusy(true);
    try {
      await api.post(`/sessions/${id}/fail`, { reason });
      await api.post(`/sessions/${id}/retry`);
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      setBlob(null);
      setBlobUrl(null);
      setConfirmDiscard(false);
      setPhotoUrl(null);
      setMicHeard(false);
      setTrackLost(null);
      setPhase("check");
      sess.reload();
      toast("Ready to record again.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }

  // --- render ------------------------------------------------------------------------------------
  if (sess.loading && !sess.data) return <Loading />;
  if (sess.error || !sess.data) return <ErrorNotice message={sess.error ?? "Not found"} onRetry={sess.reload} />;
  const s = sess.data;
  const script = scripts.data?.scripts[s.script_id ?? ""];
  const minS = scripts.data?.min_duration_s ?? 1200;
  const header = (
    <PageHeader
      crumbs={<><Link to="/sessions">Sessions</Link> / <Link to={`/sessions/${s.id}`}>{s.id}</Link></>}
      title={phase === "recording" ? "Recording" : phase === "review" ? "Review and upload" : "Check equipment"}
      description={`${s.participant_code}, ${script?.title ?? s.script_id}`}
    />
  );

  // states where this page cannot continue
  if (phase === "check" && s.status !== "CONSENTED") {
    const interrupted = s.status === "RECORDING" || s.status === "RECORDED";
    return (
      <>
        {header}
        <Panel>
          {interrupted ? (
            <>
              <div className="notice notice-warn" style={{ marginBottom: 14 }}>
                <strong>This recording was interrupted.</strong>
                The page was closed or reloaded before the recording was uploaded, so the video in the browser was lost.
              </div>
              <button className="btn btn-primary" disabled={busy} onClick={() => discard("recording interrupted before upload")}>
                Record this session again
              </button>
            </>
          ) : s.status === "FAILED" ? (
            <>
              <p>This session is marked as failed.</p>
              <button className="btn btn-primary" disabled={busy} onClick={async () => { await api.post(`/sessions/${id}/retry`); sess.reload(); }}>
                Record again
              </button>
            </>
          ) : s.status === "CREATED" ? (
            <p>Consent has not been confirmed for this session yet. <Link to={`/sessions/${s.id}`}>Open the session</Link> to continue.</p>
          ) : (
            <p>This session has already been recorded. <Link to={`/sessions/${s.id}`}>Open the session</Link>.</p>
          )}
        </Panel>
      </>
    );
  }

  const activeCue = phase === "recording"
    ? episodes.find((e) => elapsed >= e.scheduled_start_ms && (e.scheduled_end_ms > 0 ? elapsed < e.scheduled_end_ms : elapsed < e.scheduled_start_ms + CUE_SHOW_MS_WHEN_OPEN_ENDED))
    : undefined;
  const nextCue = phase === "recording" ? episodes.find((e) => e.scheduled_start_ms > elapsed) : undefined;
  const scriptLane = {
    label: "Script cues",
    color: "#c98a2b",
    intervals: episodes.map((e) => ({
      start: e.scheduled_start_ms,
      end: e.scheduled_end_ms > 0 ? e.scheduled_end_ms : Math.max(elapsed, minS * 1000),
      title: VIOLATION_LABEL[e.violation_type],
    })),
  };
  const timelineMs = Math.max(elapsed + 30000, script?.rehearsal ? 60000 : minS * 1000, ...episodes.map((e) => e.scheduled_end_ms + 30000));

  return (
    <>
      {header}
      <Steps current={phase === "check" ? 2 : phase === "recording" ? 3 : 4} />

      {phase === "check" && (
        <div className="console">
          <div>
            <div className="video-frame">
              <video ref={videoRef} muted playsInline aria-label="Camera preview" />
              {!stream && (
                <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#c3cbd4", textAlign: "center", padding: 20 }}>
                  <div>
                    <p>The camera is off.</p>
                    <button className="btn btn-primary" onClick={() => openDevices()}>Turn on camera and microphone</button>
                  </div>
                </div>
              )}
            </div>
            {deviceError && <div className="notice notice-bad" style={{ marginTop: 12 }}>{deviceError}</div>}
            {stream && (
              <div className="grid-2" style={{ marginTop: 14, gap: 12 }}>
                <label className="field">
                  <span className="label">Camera</span>
                  <select className="select" value={camId} onChange={(e) => openDevices(e.target.value, micId)}>
                    {devices.cams.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span className="label">Microphone</span>
                  <select className="select" value={micId} onChange={(e) => openDevices(camId, e.target.value)}>
                    {devices.mics.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Microphone ${i + 1}`}</option>)}
                  </select>
                </label>
              </div>
            )}
          </div>
          <div className="stack">
            <Panel title="Before you start">
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 14 }}>
                <li>
                  <div className="row"><Badge tone={stream ? "ok" : "neutral"}>{stream ? "Camera on" : "Camera off"}</Badge>
                    {resolution && <span className="small muted num">{resolution.w}×{resolution.h}</span>}</div>
                </li>
                <li>
                  <div className="row" style={{ marginBottom: 6 }}>
                    <Badge tone={micHeard ? "ok" : "neutral"}>{micHeard ? "Microphone working" : "Say a few words"}</Badge>
                  </div>
                  <div className="meter" aria-label="Microphone level"><span style={{ width: `${level * 100}%` }} /></div>
                </li>
                <li>
                  <div className="row" style={{ marginBottom: 6 }}>
                    <Badge tone={photoUrl ? "ok" : "neutral"}>{photoUrl ? "Enrolment photo saved" : "Enrolment photo needed"}</Badge>
                  </div>
                  <p className="small muted" style={{ margin: "0 0 8px" }}>
                    The registered participant faces the camera. This photo is the identity reference for the session.
                  </p>
                  <div className="row">
                    <button className="btn btn-sm" disabled={!stream || photoBusy} onClick={takePhoto}>
                      {photoUrl ? "Retake photo" : "Take photo"}
                    </button>
                    {photoUrl && <img src={photoUrl} alt="Enrolment" style={{ height: 48, borderRadius: 4 }} />}
                  </div>
                </li>
                <li>
                  <label className="check" style={{ padding: 0 }}>
                    <input type="checkbox" checked={useFullscreen} onChange={(e) => setUseFullscreen(e.target.checked)} />
                    <span>Open the quiz in full screen</span>
                  </label>
                </li>
              </ul>
              {!mimeRef.current && (
                <div className="notice notice-bad" style={{ marginTop: 14 }}>
                  This browser cannot record WebM video. Use Chrome, Edge or Firefox.
                </div>
              )}
              <button
                className="btn btn-primary btn-lg"
                style={{ marginTop: 16, width: "100%" }}
                disabled={!stream || !micHeard || !photoUrl || !mimeRef.current || busy}
                onClick={start}
              >
                Start recording
              </button>
            </Panel>
            {script && script.episodes.length > 0 && (
              <div className="notice notice-info">
                <strong>{script.episodes.length} scripted {script.episodes.length === 1 ? "cue" : "cues"}.</strong>
                Instructions appear on screen at the scheduled time. Follow each one until it disappears.
              </div>
            )}
          </div>
        </div>
      )}

      {phase === "recording" && (
        <>
          {trackLost && (
            <div className="notice notice-bad" style={{ marginBottom: 16 }}>
              <strong>{trackLost}</strong> Stop the recording; you can discard it and record again.
            </div>
          )}
          <div className="console">
            <div>
              <div className="video-frame">
                <video ref={videoRef} muted playsInline aria-label="Live camera" />
                <div className="video-overlay">
                  <Badge tone="live">Recording</Badge>
                  <span className="timer">{fmtClock(elapsed)}</span>
                </div>
              </div>
              <div className="meter" style={{ marginTop: 10 }} aria-label="Microphone level"><span style={{ width: `${level * 100}%` }} /></div>
              <div className="panel" style={{ marginTop: 14 }}>
                <div className="panel-body">
                  <Timeline durationMs={timelineMs} lanes={[scriptLane]} playheadMs={elapsed} />
                  <div className="row small muted" style={{ marginTop: 8 }}>
                    <span>
                      {nextCue ? `Next cue in ${fmtClock(nextCue.scheduled_start_ms - elapsed)}` : episodes.length ? "All cues shown" : "No cues in this script"}
                    </span>
                    <span className="spacer" />
                    {tele && (
                      <span className="num" title="Browser telemetry events">
                        Telemetry: {tele.sent} sent{tele.pending > 0 ? `, ${tele.pending} waiting` : ""}
                        {tele.lastError ? " (retrying)" : ""}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="row" style={{ marginTop: 16 }}>
                <button className="btn btn-danger-solid btn-lg" disabled={busy} onClick={() => setConfirmStop(true)}>
                  Stop recording
                </button>
                {!script?.rehearsal && elapsed < minS * 1000 && (
                  <span className="small muted">Corpus sessions need at least {minS / 60} minutes ({fmtClock(minS * 1000 - elapsed)} to go).</span>
                )}
              </div>
            </div>
            <div className="stack">
              {activeCue && (
                <div className="cue" role="alert">
                  <div className="cue-title">{VIOLATION_LABEL[activeCue.violation_type] ?? "Instruction"}</div>
                  <div>{activeCue.instruction}</div>
                  {activeCue.scheduled_end_ms > 0 && (
                    <div className="small num" style={{ marginTop: 6 }}>
                      {Math.ceil((activeCue.scheduled_end_ms - elapsed) / 1000)} s remaining
                    </div>
                  )}
                </div>
              )}
              <Panel title="Practice quiz" description="Answer in your own words. Nothing here is graded.">
                {QUIZ.map((q, i) => (
                  <div className="quiz-q" key={q.id}>
                    <label htmlFor={q.id}>{i + 1}. {q.q}</label>
                    <textarea id={q.id} className="textarea" rows={2} value={answers[q.id] ?? ""}
                      onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })} />
                  </div>
                ))}
              </Panel>
            </div>
          </div>
        </>
      )}

      {phase === "review" && blob && blobUrl && (
        <div className="console">
          <div>
            <div className="video-frame">
              <video src={blobUrl} controls playsInline aria-label="Recorded session" />
            </div>
          </div>
          <div className="stack">
            <Panel title="Recording">
              <dl className="dl">
                <dt>Length</dt><dd className="num">{fmtClock(durationS * 1000)}</dd>
                <dt>Size</dt><dd className="num">{fmtBytes(blob.size)}</dd>
                <dt>Format</dt><dd>{mimeRef.current}</dd>
                <dt>Telemetry</dt><dd className="num">{tele ? `${tele.sent} events sent${tele.rejected ? `, ${tele.rejected} rejected` : ""}` : "—"}</dd>
              </dl>
              {!script?.rehearsal && durationS < minS && (
                <div className="notice notice-warn" style={{ marginTop: 14 }}>
                  Shorter than {minS / 60} minutes: it will be stored but will not count toward the corpus.
                </div>
              )}
              {uploadPct != null && (
                <div style={{ marginTop: 14 }}>
                  <div className="row small" style={{ justifyContent: "space-between", marginBottom: 4 }}>
                    <span>Uploading</span><span className="num">{Math.round(uploadPct * 100)}%</span>
                  </div>
                  <div className="progress"><span style={{ width: `${uploadPct * 100}%` }} /></div>
                </div>
              )}
              {uploadError && <div className="notice notice-bad" style={{ marginTop: 14 }}>{uploadError} Your recording is still in this tab; try again.</div>}
              <div className="row" style={{ marginTop: 16 }}>
                <button className="btn btn-primary btn-lg" disabled={uploadPct != null} onClick={upload}>Upload recording</button>
                <a className="btn" href={blobUrl} download={`${id}.webm`}>Save a local copy</a>
              </div>
              <button className="btn btn-ghost btn-sm" style={{ marginTop: 10, color: "var(--bad)" }} disabled={uploadPct != null} onClick={() => setConfirmDiscard(true)}>
                Discard and record again
              </button>
            </Panel>
          </div>
        </div>
      )}

      {confirmStop && (
        <Modal
          title="Stop recording?"
          onClose={() => setConfirmStop(false)}
          footer={<><button className="btn" onClick={() => setConfirmStop(false)}>Keep recording</button><button className="btn btn-danger-solid" onClick={stop}>Stop recording</button></>}
        >
          <p>
            {!script?.rehearsal && elapsed < minS * 1000
              ? `The session is ${fmtClock(elapsed)} long. Sessions under ${minS / 60} minutes do not count toward the corpus.`
              : "You can review the recording before uploading it."}
          </p>
        </Modal>
      )}
      {confirmDiscard && (
        <Modal
          title="Discard this recording?"
          description="The recording is deleted from this browser. The session returns to the equipment check."
          onClose={() => setConfirmDiscard(false)}
          footer={<><button className="btn" onClick={() => setConfirmDiscard(false)}>Keep it</button><button className="btn btn-danger-solid" disabled={busy} onClick={() => discard("discarded by operator before upload")}>Discard</button></>}
        />
      )}
    </>
  );
}
