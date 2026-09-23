import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Icon } from "./components/ui";
import { operatorName } from "./lib/api";
import DatasetsPage from "./pages/DatasetsPage";
import NewSessionPage from "./pages/NewSessionPage";
import OverviewPage from "./pages/OverviewPage";
import ParticipantDetailPage from "./pages/ParticipantDetailPage";
import ParticipantsPage from "./pages/ParticipantsPage";
import RecordPage from "./pages/RecordPage";
import SessionDetailPage from "./pages/SessionDetailPage";
import SessionsPage from "./pages/SessionsPage";
import SystemPage from "./pages/SystemPage";

function OperatorField() {
  const [name, setName] = useState(operatorName() === "operator" ? "" : operatorName());
  return (
    <label>
      <span className="muted">Operator name (for the audit log)</span>
      <input
        className="input operator-input"
        value={name}
        placeholder="Your name"
        maxLength={80}
        onChange={(e) => {
          setName(e.target.value);
          try {
            localStorage.setItem("rf.operator", e.target.value.trim() || "operator");
          } catch {
            /* storage unavailable: the audit log records "operator" */
          }
        }}
      />
    </label>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);
  const recording = /^\/sessions\/[^/]+\/record$/.test(loc.pathname);

  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Main navigation">
        <Link to="/" className="brand">
          <span className="brand-mark" aria-hidden>
            R
          </span>
          <span className="brand-name">RiskFusion</span>
        </Link>
        <nav className="nav">
          <NavLink to="/" end>
            <Icon name="overview" /> Overview
          </NavLink>
          <NavLink to="/sessions/new">
            <Icon name="record" /> New session
          </NavLink>
          <NavLink to="/sessions" end>
            <Icon name="sessions" /> Sessions
          </NavLink>
          <NavLink to="/participants">
            <Icon name="people" /> Participants
          </NavLink>
        </nav>
        <div className="nav-group">Data</div>
        <nav className="nav">
          <NavLink to="/datasets">
            <Icon name="data" /> Datasets
          </NavLink>
          <NavLink to="/system">
            <Icon name="system" /> System status
          </NavLink>
        </nav>
        <div className="sidebar-foot">
          <OperatorField />
        </div>
      </aside>
      <div>
        <div className="topbar">
          <button className="btn btn-ghost btn-sm" aria-label="Open navigation" onClick={() => setOpen(true)}>
            <Icon name="menu" />
          </button>
          <strong style={{ color: "var(--ink)" }}>RiskFusion</strong>
        </div>
        <main className="main" onClick={() => open && setOpen(false)}>
          <div className="page" style={recording ? { maxWidth: 1400 } : undefined}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/participants" element={<ParticipantsPage />} />
              <Route path="/participants/:id" element={<ParticipantDetailPage />} />
              <Route path="/sessions" element={<SessionsPage />} />
              <Route path="/sessions/new" element={<NewSessionPage />} />
              <Route path="/sessions/:id" element={<SessionDetailPage />} />
              <Route path="/sessions/:id/record" element={<RecordPage />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="/system" element={<SystemPage />} />
              <Route
                path="*"
                element={
                  <div className="empty">
                    <h3>This page does not exist.</h3>
                    <Link to="/">Go to the overview</Link>
                  </div>
                }
              />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
