"""
MYCONEX Web Dashboard
----------------------
FastAPI single-file dashboard with live activity feed, data submission,
and project idea browser.

Pages:
  GET  /              — status overview
  GET  /activity      — live rolling activity log (SSE)
  GET  /log           — live tail of /tmp/buzlock.log
  GET  /submit        — ingest URLs, text notes, or manage feeds
  GET  /projects      — project ideas extracted from ingested content
  GET  /profile       — full interest profile
  GET  /wisdom        — wisdom store browser
  GET  /feeds         — RSS + podcast feed manager
  GET  /signals       — cross-source signals
  GET  /feedback      — feedback log

SSE:
  GET  /stream/activity  — Server-Sent Events activity stream
  GET  /stream/log       — live tail of /tmp/buzlock.log

API:
  POST /api/submit/url   — ingest a URL into the knowledge base
  POST /api/submit/text  — save a text note into the knowledge base
  GET  /api/status       — ingester status JSON
  GET  /api/profile      — interest profile JSON
  GET  /api/wisdom       — wisdom store JSON (paginated)
  GET  /api/signals      — signals log JSON
  GET  /api/feedback     — feedback stats JSON

Feed management:
  POST /htmx/rss/add        — add RSS feed
  POST /htmx/rss/remove     — remove RSS feed
  POST /htmx/podcast/add    — add podcast feed
  POST /htmx/podcast/remove — remove podcast feed

Usage:
    cd ~/myconex && uvicorn dashboard.app:app --host 0.0.0.0 --port 7860 --reload

Or via buzlock_bot.py background task.

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.warning("[dashboard] FastAPI not installed — run: pip install fastapi uvicorn")

_BASE = Path.home() / ".myconex"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _load_jsonl(path: Path) -> list[dict]:
    lines = []
    try:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return lines


def _ingester_status() -> dict[str, Any]:
    ingesters = {}
    for name, fname in [
        ("email",   "email_insights.json"),
        ("youtube", "youtube_insights.json"),
        ("rss",     "rss_insights.json"),
        ("podcast", "podcast_insights.json"),
    ]:
        log = _load(_BASE / fname, [])
        if log:
            last = log[-1]
            ingesters[name] = {
                "total":      len(log),
                "last_run":   last.get("processed_at", ""),
                "last_title": last.get("title", last.get("subject", "")),
                "has_error":  bool(last.get("error")),
            }
        else:
            ingesters[name] = {"total": 0, "last_run": "", "last_title": "", "has_error": False}
    return ingesters


def _feedback_stats() -> dict[str, Any]:
    fb    = _load_jsonl(_BASE / "feedback_log.jsonl")
    total = len(fb)
    pos   = sum(1 for f in fb if f.get("positive"))
    return {
        "total":    total,
        "positive": pos,
        "negative": total - pos,
        "rate":     round(pos / total * 100) if total else None,
        "recent":   fb[-10:][::-1],
    }


def _sysmon_data() -> dict[str, Any]:
    """Return live system resource stats for the sidebar monitor."""
    if not _PSUTIL_OK:
        return {"available": False}

    cpu_pct  = _psutil.cpu_percent(interval=0.2)
    mem      = _psutil.virtual_memory()
    disk     = _psutil.disk_usage("/")
    net      = _psutil.net_io_counters()
    temps: dict[str, float] = {}
    try:
        raw = _psutil.sensors_temperatures()
        if raw:
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in raw and raw[key]:
                    temps["cpu"] = round(raw[key][0].current, 1)
                    break
    except Exception:
        pass

    uptime_s = int(time.time() - _psutil.boot_time())
    h, rem   = divmod(uptime_s, 3600)
    m        = rem // 60
    uptime   = f"{h}h {m}m" if h else f"{m}m"

    return {
        "available":  True,
        "cpu_pct":    round(cpu_pct, 1),
        "mem_pct":    round(mem.percent, 1),
        "mem_used_gb": round(mem.used / 1e9, 1),
        "mem_total_gb": round(mem.total / 1e9, 1),
        "disk_pct":   round(disk.percent, 1),
        "disk_used_gb": round(disk.used / 1e9, 1),
        "disk_total_gb": round(disk.total / 1e9, 1),
        "net_sent_mb": round(net.bytes_sent / 1e6, 1),
        "net_recv_mb": round(net.bytes_recv / 1e6, 1),
        "cpu_temp":   temps.get("cpu"),
        "uptime":     uptime,
        "load":       list(_psutil.getloadavg()),
    }


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """<style>
:root {
  --bg:           #07070f;
  --surface:      #0c0c1a;
  --surface2:     #111124;
  --surface3:     #161630;
  --border:       #1c1c30;
  --border-hi:    #2c2c50;
  --text:         #c0c0d8;
  --text-dim:     #52527a;
  --text-bright:  #e8e8f8;
  --accent:       #4a8fff;
  --accent-dim:   #162050;
  --teal:         #00c8a0;
  --teal-dim:     #00382c;
  --green:        #44cc88;
  --red:          #cc4466;
  --amber:        #ddaa44;
  --purple:       #aa88ff;
  --sidebar-w:    210px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; }
body {
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex;
  font-size: 14px; line-height: 1.5;
}

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-w); min-height: 100vh;
  background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  position: fixed; top: 0; left: 0; height: 100vh;
  overflow-y: auto; z-index: 20;
}
.sidebar-brand {
  padding: 22px 20px 18px; border-bottom: 1px solid var(--border);
}
.brand-name {
  font-size: 15px; font-weight: 700; color: var(--teal);
  letter-spacing: 2px; text-transform: uppercase;
}
.brand-sub {
  font-size: 10px; color: var(--text-dim);
  letter-spacing: 1px; margin-top: 3px; text-transform: uppercase;
}
.nav-section {
  padding: 14px 0 4px; font-size: 10px; font-weight: 700;
  color: var(--text-dim); letter-spacing: 1.5px; text-transform: uppercase;
  padding-left: 20px;
}
.nav-link {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 20px; color: var(--text-dim); text-decoration: none;
  font-size: 13px; transition: color .12s, background .12s;
  border-left: 2px solid transparent;
}
.nav-link:hover { color: var(--text); background: var(--surface2); }
.nav-link.active {
  color: var(--accent); background: var(--accent-dim);
  border-left-color: var(--accent);
}
.nav-icon { font-size: 13px; width: 18px; text-align: center; flex-shrink: 0; }

/* ── Watch Mode ── */
.watch-btn {
  display: flex; align-items: center; gap: 8px;
  margin: 12px 14px 6px; padding: 8px 14px;
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text-dim); border-radius: 6px; cursor: pointer;
  font-size: 12px; font-family: inherit;
  transition: color .12s, background .12s, border-color .12s;
  width: calc(100% - 28px);
}
.watch-btn:hover { color: var(--text); background: var(--surface); }
.watch-btn.on {
  color: var(--teal); border-color: var(--teal);
  background: var(--teal-dim);
}
.watch-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: currentColor; flex-shrink: 0;
}
.watch-btn.on .watch-dot { animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.watch-countdown {
  margin-left: auto; font-size: 10px; opacity: .7; font-variant-numeric: tabular-nums;
}
.sidebar-footer {
  padding-bottom: 12px; border-top: 1px solid var(--border); padding-top: 8px;
}

/* ── System Monitor ── */
.sysmon {
  border-top: 1px solid var(--border);
  padding: 14px 16px 10px;
  margin-top: auto;
  flex-shrink: 0;
}
.sysmon-title {
  font-size: 9px; font-weight: 700; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--text-dim);
  margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}
.sysmon-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); flex-shrink: 0;
  animation: pulse 3s ease-in-out infinite;
}
.sysmon-row {
  display: flex; align-items: center; gap: 6px;
  margin-bottom: 7px; min-width: 0;
}
.sysmon-label {
  font-size: 10px; color: var(--text-dim); width: 34px;
  flex-shrink: 0; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px;
}
.sysmon-bar-wrap {
  flex: 1; background: var(--surface3); border-radius: 3px;
  height: 5px; overflow: hidden; min-width: 0;
}
.sysmon-bar {
  height: 100%; border-radius: 3px;
  transition: width .6s ease, background-color .6s ease;
  background: var(--accent);
}
.sysmon-bar.warn  { background: var(--amber); }
.sysmon-bar.crit  { background: var(--red); }
.sysmon-val {
  font-size: 10px; color: var(--text); font-variant-numeric: tabular-nums;
  width: 32px; text-align: right; flex-shrink: 0;
  white-space: nowrap;
}
.sysmon-extra {
  display: flex; justify-content: space-between;
  font-size: 10px; color: var(--text-dim);
  margin-top: 6px; padding-top: 6px;
  border-top: 1px solid var(--border);
}
.sysmon-extra span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sysmon-na {
  font-size: 10px; color: var(--text-dim); font-style: italic;
  text-align: center; padding: 6px 0;
}

/* ── Main ── */
.main {
  margin-left: var(--sidebar-w); flex: 1;
  padding: 30px 36px; max-width: calc(100vw - var(--sidebar-w));
  min-height: 100vh;
}
.page-header {
  display: flex; align-items: baseline; gap: 12px;
  margin-bottom: 24px; padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.page-title { font-size: 20px; font-weight: 600; color: var(--text-bright); }
.page-sub { font-size: 13px; color: var(--text-dim); }

/* ── Stat cards ── */
.card-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 28px;
}
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px;
  transition: border-color .15s;
}
.card:hover { border-color: var(--border-hi); }
.card.ok    { border-color: #14382a; }
.card.warn  { border-color: #3a2a10; }
.card.error { border-color: #3a0e1e; }
.card-label {
  font-size: 10px; font-weight: 700; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.card-value { font-size: 30px; font-weight: 700; color: var(--text-bright); line-height: 1; }
.card-sub { font-size: 11px; color: var(--text-dim); margin-top: 6px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ── Section headers ── */
h2 {
  font-size: 12px; font-weight: 700; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 1px;
  margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}

/* ── Tables ── */
.tbl {
  width: 100%; border-collapse: collapse; font-size: 13px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden;
}
.tbl th {
  background: var(--surface2); color: var(--text-dim);
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .8px; padding: 11px 14px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.tbl td { padding: 10px 14px; border-top: 1px solid var(--border); vertical-align: top; }
.tbl tr:hover td { background: var(--surface2); }

/* ── Tags ── */
.tag {
  display: inline-block; background: var(--accent-dim); color: var(--accent);
  font-size: 11px; padding: 2px 8px; border-radius: 4px; margin: 2px;
}

/* ── Buttons ── */
.btn {
  background: var(--accent-dim); color: var(--accent);
  border: 1px solid var(--accent); border-radius: 8px;
  padding: 9px 18px; cursor: pointer; font-size: 13px; font-weight: 500;
  transition: background .12s; font-family: inherit; white-space: nowrap;
}
.btn:hover { background: #1e3a80; }
.btn-teal { background: var(--teal-dim); color: var(--teal); border-color: var(--teal); }
.btn-teal:hover { background: #005048; }
.btn-sm { padding: 6px 12px; font-size: 11px; border-radius: 6px; }
.btn-danger { background: #280a18; color: var(--red); border-color: var(--red); }
.btn-danger:hover { background: #3a1020; }

/* ── Forms ── */
.form-block {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 22px;
}
.form-row { display: flex; gap: 10px; margin-bottom: 12px; }
.form-row input, .form-row textarea { flex: 1; }
.form-label {
  font-size: 11px; font-weight: 700; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: .8px;
  display: block; margin-bottom: 6px;
}
input[type=text], input[type=url], textarea, select {
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 10px 14px; border-radius: 8px;
  font-size: 13px; width: 100%; outline: none;
  transition: border-color .15s; font-family: inherit;
}
input:focus, textarea:focus, select:focus { border-color: var(--accent); }
textarea { resize: vertical; min-height: 110px; line-height: 1.5; }
.form-hint { font-size: 11px; color: var(--text-dim); margin-top: 6px; }

/* ── Tabs ── */
.tab-bar {
  display: flex; gap: 3px; background: var(--surface2);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 4px; margin-bottom: 20px;
}
.tab {
  flex: 1; padding: 9px 12px; text-align: center; font-size: 13px;
  font-weight: 500; cursor: pointer; border-radius: 7px;
  color: var(--text-dim); transition: all .12s;
  border: none; background: none; font-family: inherit;
}
.tab.active {
  background: var(--surface); color: var(--text-bright);
  box-shadow: 0 1px 6px rgba(0,0,0,.5);
}
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Submit result feedback ── */
.submit-result {
  display: none; padding: 10px 14px; border-radius: 8px;
  font-size: 13px; margin-top: 12px; font-weight: 500;
}
.submit-result.ok  { background: #0e2a1a; color: var(--green); border: 1px solid #1a4a2a; }
.submit-result.err { background: #280a14; color: var(--red);   border: 1px solid #4a1428; }

/* ── Activity wall ── */
.activity-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px;
}
.conn-status {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--text-dim);
}
.conn-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim);
  flex-shrink: 0; transition: background .3s;
}
.conn-dot.live { background: var(--green); animation: pulse 2.5s ease-in-out infinite; }
.conn-dot.dead { background: var(--red); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.activity-wall {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; height: 520px; overflow-y: auto;
  font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px; padding: 14px 16px; scroll-behavior: smooth;
}
.activity-wall:empty::before {
  content: 'Waiting for activity...';
  color: var(--text-dim); font-style: italic;
}
.log-line { padding: 1px 0; line-height: 1.65; }
.log-ts { color: var(--text-dim); margin-right: 12px; user-select: none; }
.log-info    { color: var(--teal); }
.log-success { color: var(--green); }
.log-warn    { color: var(--amber); }
.log-error   { color: var(--red); }
.log-default { color: var(--text); }

/* ── Project cards ── */
.project-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 14px;
}
details.project-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px;
  transition: border-color .15s, box-shadow .15s;
  cursor: pointer;
}
details.project-card:hover { border-color: var(--border-hi); }
details.project-card[open] {
  border-color: var(--border-hi);
  box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
details.project-card summary { list-style: none; outline: none; }
details.project-card summary::-webkit-details-marker { display: none; }
.proj-num {
  font-size: 10px; font-weight: 700; color: var(--text-dim);
  letter-spacing: .8px; margin-bottom: 8px; text-transform: uppercase;
}
.proj-badge {
  display: inline-block; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .6px;
  padding: 2px 8px; border-radius: 4px; margin-bottom: 10px;
}
.pb-0 { background: #12204a; color: #6aabff; }
.pb-1 { background: #1e103a; color: #bb99ff; }
.pb-2 { background: #00281e; color: #44ddaa; }
.pb-3 { background: #2e1a00; color: #ddaa44; }
.proj-text {
  font-size: 13px; color: var(--text); line-height: 1.55;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden;
}
details.project-card[open] .proj-text {
  display: block; -webkit-line-clamp: unset; overflow: visible;
}
.proj-expand {
  font-size: 11px; color: var(--text-dim); margin-top: 10px;
}
details.project-card[open] .proj-expand { display: none; }

/* ── Signal cards ── */
.signal-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;
}
.signal-card:hover { border-color: var(--border-hi); }
.signal-topic { color: var(--teal); font-size: 15px; font-weight: 600; margin-bottom: 8px; }
.signal-meta { margin-bottom: 6px; font-size: 12px; color: var(--text-dim); }
.signal-snippet { color: var(--text-dim); font-size: 12px; line-height: 1.5; }

/* ── Bar chart ── */
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.bar-label { font-size: 12px; color: var(--text); min-width: 160px; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; background: var(--surface2); border-radius: 4px; height: 8px;
             border: 1px solid var(--border); }
.bar-fill { height: 100%; border-radius: 4px; background: var(--accent); min-width: 2px; }
.bar-count { font-size: 11px; color: var(--text-dim); min-width: 30px; text-align: right; }

/* ── Misc ── */
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.badge-ok  { background: #0e2a1a; color: var(--green); }
.badge-warn{ background: #2e1e00; color: var(--amber); }
.badge-err { background: #280a14; color: var(--red); }
.pos { color: var(--green); } .neg { color: var(--red); }
.empty { color: var(--text-dim); font-size: 13px; padding: 40px; text-align: center; font-style: italic; }
.gap { margin-bottom: 24px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

@media (max-width: 680px) {
  .sidebar { display: none; }
  .main { margin-left: 0; padding: 16px; }
}
</style>"""


# ── JavaScript ─────────────────────────────────────────────────────────────────

_JS = """<script>
// ── Activity stream ───────────────────────────────────────────────────────────
function initActivityStream() {
  const wall = document.getElementById('activity-wall');
  if (!wall) return;
  const dot    = document.getElementById('conn-dot');
  const status = document.getElementById('conn-status-text');
  let paused   = false;

  wall.addEventListener('mouseenter', () => { paused = true; });
  wall.addEventListener('mouseleave', () => {
    paused = false;
    wall.scrollTop = wall.scrollHeight;
  });

  function esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function classify(text) {
    const t = text.toLowerCase();
    if (/error|fail|crash|exception|traceback/.test(t)) return 'error';
    if (/warn/.test(t))                                  return 'warn';
    if (/✔|done|complete|success|added|processed|saved|ingested/.test(t)) return 'success';
    if (/start|connect|init|load|launch/.test(t))        return 'info';
    return 'default';
  }

  function addLine(text, cls) {
    const line = document.createElement('div');
    line.className = 'log-line';
    const ts = new Date().toLocaleTimeString('en-US', {hour12: false, timeZoneName: undefined});
    line.innerHTML = '<span class="log-ts">' + ts + '</span>'
                   + '<span class="log-' + cls + '">' + esc(text) + '</span>';
    wall.appendChild(line);
    if (!paused) wall.scrollTop = wall.scrollHeight;
    while (wall.children.length > 600) wall.removeChild(wall.firstChild);
  }

  function connect() {
    dot.className   = 'conn-dot';
    status.textContent = 'connecting…';
    const es = new EventSource('/stream/activity');

    es.onopen = () => {
      dot.className      = 'conn-dot live';
      status.textContent = 'live';
    };

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'ping') return;
        const text = data.text || String(data);
        addLine(text, classify(text));
      } catch {
        if (e.data && e.data !== '{"type":"ping"}') addLine(e.data, 'default');
      }
    };

    es.onerror = () => {
      dot.className      = 'conn-dot dead';
      status.textContent = 'reconnecting in 5s…';
      es.close();
      setTimeout(connect, 5000);
    };
  }

  connect();
}

// ── Bot log stream ────────────────────────────────────────────────────────────
function initLogStream() {
  const wall = document.getElementById('log-wall');
  if (!wall) return;
  const dot    = document.getElementById('log-conn-dot');
  const status = document.getElementById('log-conn-status-text');
  let paused   = false;

  wall.addEventListener('mouseenter', () => { paused = true; });
  wall.addEventListener('mouseleave', () => {
    paused = false;
    wall.scrollTop = wall.scrollHeight;
  });

  function esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function classifyLog(text) {
    if (/ERROR|error|CRITICAL|Traceback|Exception/.test(text)) return 'error';
    if (/WARNING|WARN/.test(text))  return 'warn';
    if (/INFO.*connect|INFO.*start|INFO.*online|INFO.*loaded/.test(text)) return 'info';
    if (/INFO.*process|INFO.*stored|INFO.*done|INFO.*success/.test(text)) return 'success';
    return 'default';
  }

  function addLine(text) {
    const line = document.createElement('div');
    line.className = 'log-line';
    // Extract timestamp from log line if present (e.g. "2026-03-25 15:54:09,886")
    const m = text.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\d]*)\s+\[?\w+\]?\s*/);
    let ts = '', rest = text;
    if (m) { ts = m[1].slice(11, 19); rest = text.slice(m[0].length); }
    const cls = classifyLog(text);
    line.innerHTML = (ts ? '<span class="log-ts">' + esc(ts) + '</span>' : '')
                   + '<span class="log-' + cls + '">' + esc(rest || text) + '</span>';
    wall.appendChild(line);
    if (!paused) wall.scrollTop = wall.scrollHeight;
    while (wall.children.length > 1000) wall.removeChild(wall.firstChild);
  }

  function connect() {
    dot.className = 'conn-dot';
    status.textContent = 'connecting…';
    const es = new EventSource('/stream/log');

    es.onopen = () => {
      dot.className = 'conn-dot live';
      status.textContent = 'live';
    };

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'ping') return;
        addLine(data.line || String(data));
      } catch {
        if (e.data && e.data !== '{"type":"ping"}') addLine(e.data);
      }
    };

    es.onerror = () => {
      dot.className = 'conn-dot dead';
      status.textContent = 'reconnecting in 5s…';
      es.close();
      setTimeout(connect, 5000);
    };
  }

  connect();
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function initTabs(defaultTab) {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const group = tab.closest('.tab-group');
      group.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      group.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
  if (defaultTab) {
    const btn = document.querySelector('[data-tab="' + defaultTab + '"]');
    if (btn) btn.click();
  }
}

// ── Data submission ───────────────────────────────────────────────────────────
async function submitData(endpoint, payload, resultId) {
  const result = document.getElementById(resultId);
  result.className = 'submit-result';
  result.style.display = 'none';
  try {
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    result.textContent   = data.message || (r.ok ? 'Done.' : 'Error.');
    result.className     = 'submit-result ' + (r.ok && data.ok !== false ? 'ok' : 'err');
    result.style.display = 'block';
  } catch (e) {
    result.textContent   = 'Network error: ' + e;
    result.className     = 'submit-result err';
    result.style.display = 'block';
  }
}

function submitUrl() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;
  submitData('/api/submit/url', {url}, 'url-result');
}

function submitText() {
  const text  = document.getElementById('text-input').value.trim();
  const title = document.getElementById('text-title').value.trim() || 'Note';
  if (!text) return;
  submitData('/api/submit/text', {text, title}, 'text-result');
}

// ── Watch Mode ───────────────────────────────────────────────────────────────
const _WATCH_PAGES = ['/', '/activity', '/log', '/submit', '/projects',
                      '/profile', '/wisdom', '/feeds', '/signals', '/feedback'];
const _WATCH_INTERVAL = 60;  // seconds between page changes

let _watchTimer  = null;
let _watchRemain = _WATCH_INTERVAL;
let _watchTick   = null;

function _watchCurrentIdx() {
  const path = location.pathname;
  const idx  = _WATCH_PAGES.indexOf(path);
  return idx === -1 ? 0 : idx;
}

function _watchUpdateBtn(on) {
  const btn       = document.getElementById('watch-btn');
  const countdown = document.getElementById('watch-countdown');
  if (!btn) return;
  btn.className = 'watch-btn' + (on ? ' on' : '');
  btn.querySelector('.watch-label').textContent = on ? 'Watch ON' : 'Watch Mode';
  if (countdown) countdown.style.display = on ? 'inline' : 'none';
}

function _watchStop() {
  clearTimeout(_watchTimer);
  clearInterval(_watchTick);
  _watchTimer = _watchTick = null;
  localStorage.removeItem('watchMode');
  _watchUpdateBtn(false);
}

function _watchStart() {
  _watchRemain = _WATCH_INTERVAL;
  _watchUpdateBtn(true);

  _watchTick = setInterval(() => {
    _watchRemain--;
    const cd = document.getElementById('watch-countdown');
    if (cd) cd.textContent = _watchRemain + 's';
    if (_watchRemain <= 0) {
      clearInterval(_watchTick);
      const next = (_watchCurrentIdx() + 1) % _WATCH_PAGES.length;
      location.href = _WATCH_PAGES[next];
    }
  }, 1000);
}

function toggleWatchMode() {
  if (localStorage.getItem('watchMode') === '1') {
    _watchStop();
  } else {
    localStorage.setItem('watchMode', '1');
    _watchStart();
  }
}

// ── System Monitor ────────────────────────────────────────────────────────────
function _barClass(pct) {
  if (pct >= 90) return 'crit';
  if (pct >= 70) return 'warn';
  return '';
}

function _setBar(id, pct) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.width = Math.min(pct, 100) + '%';
  el.className   = 'sysmon-bar ' + _barClass(pct);
}

function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function refreshSysmon() {
  try {
    const r = await fetch('/api/sysmon');
    if (!r.ok) return;
    const d = await r.json();
    if (!d.available) return;

    _setBar  ('sm-cpu-bar',  d.cpu_pct);
    _setText ('sm-cpu-val',  d.cpu_pct + '%');

    _setBar  ('sm-mem-bar',  d.mem_pct);
    _setText ('sm-mem-val',  d.mem_pct + '%');

    _setBar  ('sm-disk-bar', d.disk_pct);
    _setText ('sm-disk-val', d.disk_pct + '%');

    const tempStr = d.cpu_temp != null ? d.cpu_temp + '°' : '—';
    _setText('sm-temp', tempStr);
    _setText('sm-uptime', d.uptime || '—');
    _setText('sm-net', '↑' + d.net_sent_mb + ' ↓' + d.net_recv_mb + ' MB');

    // Colour temp indicator
    const tempEl = document.getElementById('sm-temp');
    if (tempEl && d.cpu_temp != null) {
      tempEl.style.color = d.cpu_temp >= 85 ? 'var(--red)'
                         : d.cpu_temp >= 70 ? 'var(--amber)'
                         : 'var(--text-dim)';
    }
  } catch (_) {}
}

function initSysmon() {
  if (!document.getElementById('sm-cpu-bar')) return;
  refreshSysmon();
  setInterval(refreshSysmon, 3000);
}

// On every page load: resume watch mode + start sysmon
document.addEventListener('DOMContentLoaded', () => {
  const cd = document.getElementById('watch-countdown');
  if (localStorage.getItem('watchMode') === '1') {
    if (cd) cd.textContent = _WATCH_INTERVAL + 's';
    _watchStart();
  } else {
    _watchUpdateBtn(false);
    if (cd) cd.style.display = 'none';
  }
  initSysmon();
});
</script>"""


# ── Navigation ─────────────────────────────────────────────────────────────────

_NAV_ITEMS = [
    ("/",            "⊙", "Status",      "home"),
    ("/chat",        "◈", "Chat",        "chat"),
    ("/activity",    "▣", "Activity",    "activity"),
    ("/log",         "⬡", "Bot Log",     "log"),
    ("/submit",      "⊕", "Submit",      "submit"),
    ("/projects",    "◆", "Projects",    "projects"),
    ("/profile",     "◎", "Profile",     "profile"),
    ("/wisdom",      "✦", "Wisdom",      "wisdom"),
    ("/feeds",       "⚙", "Feeds",       "feeds"),
    ("/signals",     "⚗", "Signals",     "signals"),
    ("/feedback",    "⚖", "Feedback",    "feedback"),
    ("/usage",       "◉", "Usage",       "usage"),
    ("/bookmarklet", "⊞", "Bookmarklet", "bookmarklet"),
    ("/gcpdot",      "●", "GCP Dot",     "gcpdot"),
]


def _sidebar(active: str = "") -> str:
    links = ""
    for href, icon, label, page_id in _NAV_ITEMS:
        cls = "nav-link active" if active == page_id else "nav-link"
        links += (
            f'<a href="{href}" class="{cls}">'
            f'<span class="nav-icon">{icon}</span>{label}</a>\n'
        )
    sysmon_html = (
        '<div class="sysmon">'
          '<div class="sysmon-title">'
            '<span class="sysmon-dot"></span>'
            'System'
          '</div>'
          '<div class="sysmon-row">'
            '<span class="sysmon-label">CPU</span>'
            '<div class="sysmon-bar-wrap"><div class="sysmon-bar" id="sm-cpu-bar" style="width:0%"></div></div>'
            '<span class="sysmon-val" id="sm-cpu-val">—</span>'
          '</div>'
          '<div class="sysmon-row">'
            '<span class="sysmon-label">RAM</span>'
            '<div class="sysmon-bar-wrap"><div class="sysmon-bar" id="sm-mem-bar" style="width:0%"></div></div>'
            '<span class="sysmon-val" id="sm-mem-val">—</span>'
          '</div>'
          '<div class="sysmon-row">'
            '<span class="sysmon-label">Disk</span>'
            '<div class="sysmon-bar-wrap"><div class="sysmon-bar" id="sm-disk-bar" style="width:0%"></div></div>'
            '<span class="sysmon-val" id="sm-disk-val">—</span>'
          '</div>'
          '<div class="sysmon-extra">'
            '<span title="CPU temperature" id="sm-temp">—</span>'
            '<span title="Network I/O" id="sm-net" style="flex:1;text-align:center">—</span>'
            '<span title="Uptime" id="sm-uptime">—</span>'
          '</div>'
        '</div>'
    )

    return (
        '<nav class="sidebar">'
        '<div class="sidebar-brand">'
        '<div class="brand-name">Myconex</div>'
        '<div class="brand-sub">Knowledge Mesh</div>'
        '</div>'
        f'{links}'
        f'{sysmon_html}'
        '<div class="sidebar-footer">'
        '<button id="watch-btn" class="watch-btn" onclick="toggleWatchMode()">'
        '<span class="watch-dot"></span>'
        '<span class="watch-label">Watch Mode</span>'
        '<span id="watch-countdown" class="watch-countdown" style="display:none">60s</span>'
        '</button>'
        '</div>'
        '</nav>'
    )


def _page(title: str, body: str, active: str = "") -> str:
    return (
        f'<!doctype html><html lang="en"><head>'
        f'<meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="theme-color" content="#07070f">'
        f'<meta name="apple-mobile-web-app-capable" content="yes">'
        f'<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
        f'<link rel="manifest" href="/manifest.json">'
        f'<title>{title} — MYCONEX</title>'
        f'{_CSS}{_JS}'
        f'</head>'
        f'<body>'
        f'{_sidebar(active)}'
        f'<div class="main">{body}</div>'
        f'<script>if("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");</script>'
        f'</body></html>'
    )


# ── Page builders ──────────────────────────────────────────────────────────────

def _status_cards_html() -> str:
    ingesters = _ingester_status()
    profile   = _load(_BASE / "interest_profile.json", {})
    fb        = _feedback_stats()
    icons     = {"email": "✉", "youtube": "▶", "rss": "◈", "podcast": "◉"}

    cards = ""
    for name, s in ingesters.items():
        cls  = "error" if s["has_error"] else ("ok" if s["total"] else "warn")
        last = s["last_run"][:10] if s["last_run"] else "never"
        cards += (
            f'<div class="card {cls}">'
            f'<div class="card-label">{icons.get(name, "")} {name}</div>'
            f'<div class="card-value">{s["total"]}</div>'
            f'<div class="card-sub">last: {last}</div>'
            f'<div class="card-sub" title="{_escape(s["last_title"])}">'
            f'{_escape(s["last_title"][:38]) or "—"}</div>'
            f'</div>'
        )

    kc = (profile.get("email_count", 0) + profile.get("video_count", 0)
          + profile.get("rss_count", 0) + profile.get("podcast_count", 0))
    cards += (
        f'<div class="card ok">'
        f'<div class="card-label">⊛ Knowledge Base</div>'
        f'<div class="card-value">{kc}</div>'
        f'<div class="card-sub">items embedded</div>'
        f'</div>'
        f'<div class="card">'
        f'<div class="card-label">⚖ Feedback</div>'
        f'<div class="card-value">{fb["total"]}</div>'
        f'<div class="card-sub">+{fb["positive"]}  −{fb["negative"]}</div>'
        f'</div>'
    )
    return cards


def _topics_html() -> str:
    profile = _load(_BASE / "interest_profile.json", {})
    topics  = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])[:20]
    if not topics:
        return '<div class="empty">No topics yet.</div>'
    max_v = topics[0][1] or 1
    rows  = ""
    for topic, count in topics:
        pct = int(count / max_v * 100)
        rows += (
            f'<div class="bar-row">'
            f'<div class="bar-label" title="{_escape(topic)}">{_escape(topic[:32])}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>'
            f'<div class="bar-count">{count}</div>'
            f'</div>'
        )
    return rows


# ── FastAPI app ────────────────────────────────────────────────────────────────

if _FASTAPI_OK:
    app = FastAPI(title="MYCONEX Dashboard", docs_url=None, redoc_url=None)

    # ── Pages ─────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def page_home() -> str:
        cards  = _status_cards_html()
        topics = _topics_html()

        signals = _load(_BASE / "signals_log.json", [])[-3:][::-1]
        sig_html = ""
        for s in signals:
            src_tags = "".join(f'<span class="tag">{_escape(src)}</span>'
                               for src in s.get("sources", []))
            sig_html += (
                f'<div class="signal-card">'
                f'<div class="signal-topic">◎ {_escape(s.get("topic", ""))}</div>'
                f'<div class="signal-meta">{src_tags}</div>'
                f'<div class="signal-snippet">{_escape(s.get("top_hit", "")[:160])}</div>'
                f'</div>'
            )

        profile = _load(_BASE / "interest_profile.json", {})
        updated = profile.get("last_updated", "")[:10] or "never"

        body = (
            f'<div class="page-header">'
            f'<div class="page-title">Status</div>'
            f'<div class="page-sub">Updated {updated}</div>'
            f'</div>'
            f'<div class="card-grid">{cards}</div>'
            f'<h2>Top Topics</h2>'
            f'<div class="gap">{topics}</div>'
            + (f'<h2>Recent Signals</h2>{sig_html}' if sig_html else '')
        )
        return _page("Status", body, "home")

    @app.get("/activity", response_class=HTMLResponse)
    async def page_activity() -> str:
        body = (
            '<div class="page-header">'
            '<div class="page-title">Activity</div>'
            '<div class="page-sub">Live event stream from all ingesters and agents</div>'
            '</div>'
            '<div class="activity-header">'
            '<div class="conn-status">'
            '<span class="conn-dot" id="conn-dot"></span>'
            '<span id="conn-status-text">connecting…</span>'
            '</div>'
            '<span style="font-size:11px;color:var(--text-dim)">hover to pause · scroll resumes on leave</span>'
            '</div>'
            '<div class="activity-wall" id="activity-wall"></div>'
            '<script>document.addEventListener("DOMContentLoaded", initActivityStream);</script>'
        )
        return _page("Activity", body, "activity")

    @app.get("/log", response_class=HTMLResponse)
    async def page_log() -> str:
        body = (
            '<div class="page-header">'
            '<div class="page-title">Bot Log</div>'
            '<div class="page-sub">Live tail of /tmp/buzlock.log</div>'
            '</div>'
            '<div class="activity-header">'
            '<div class="conn-status">'
            '<span class="conn-dot" id="log-conn-dot"></span>'
            '<span id="log-conn-status-text">connecting…</span>'
            '</div>'
            '<span style="font-size:11px;color:var(--text-dim)">hover to pause · last 200 lines on connect</span>'
            '</div>'
            '<div class="activity-wall" id="log-wall" style="height:calc(100vh - 180px)"></div>'
            '<script>document.addEventListener("DOMContentLoaded", initLogStream);</script>'
        )
        return _page("Bot Log", body, "log")

    @app.get("/submit", response_class=HTMLResponse)
    async def page_submit() -> str:
        body = (
            '<div class="page-header">'
            '<div class="page-title">Submit</div>'
            '<div class="page-sub">Add content to the knowledge base</div>'
            '</div>'
            '<div class="tab-group">'
            '<div class="tab-bar">'
            '<button class="tab active" data-tab="tab-url">URL</button>'
            '<button class="tab" data-tab="tab-text">Text / Note</button>'
            '<button class="tab" data-tab="tab-doc">Document</button>'
            '</div>'

            '<div class="tab-content active" id="tab-url">'
            '<div class="form-block">'
            '<label class="form-label" for="url-input">Web URL</label>'
            '<div class="form-row">'
            '<input type="url" id="url-input" placeholder="https://example.com/article" autocomplete="off">'
            '<button class="btn btn-teal" onclick="submitUrl()">Ingest</button>'
            '</div>'
            '<div class="form-hint">Fetches the page, strips markup, and embeds the text into the knowledge base.</div>'
            '<div class="submit-result" id="url-result"></div>'
            '</div>'
            '</div>'

            '<div class="tab-content" id="tab-text">'
            '<div class="form-block">'
            '<label class="form-label" for="text-title">Title</label>'
            '<input type="text" id="text-title" placeholder="Optional title" style="margin-bottom:12px">'
            '<label class="form-label" for="text-input">Content</label>'
            '<textarea id="text-input" placeholder="Paste or type any text, notes, ideas…"></textarea>'
            '<div style="margin-top:12px">'
            '<button class="btn btn-teal" onclick="submitText()">Save Note</button>'
            '</div>'
            '<div class="submit-result" id="text-result"></div>'
            '</div>'
            '</div>'

            '<div class="tab-content" id="tab-doc">'
            '<div class="form-block">'
            '<label class="form-label" for="doc-title">Title (optional)</label>'
            '<input type="text" id="doc-title" placeholder="My document" style="margin-bottom:12px">'
            '<label class="form-label" for="doc-file">File</label>'
            '<input type="file" id="doc-file" accept=".pdf,.epub,.txt,.md,.rst,.csv" '
            'style="background:var(--surface2);border:1px solid var(--border);color:var(--text);'
            'padding:10px 14px;border-radius:8px;font-size:13px;width:100%;margin-bottom:12px">'
            '<div class="form-hint">Supported: PDF, EPUB, TXT, MD, RST, CSV — up to 20 chunks per document.</div>'
            '<div style="margin-top:12px">'
            '<button class="btn btn-teal" onclick="submitDoc()">Ingest Document</button>'
            '</div>'
            '<div class="submit-result" id="doc-result"></div>'
            '</div>'
            '</div>'
            '</div>'

            '<script>document.addEventListener("DOMContentLoaded", () => initTabs("tab-url"));</script>'
            '<script>'
            'async function submitDoc() {'
            '  const fileInput = document.getElementById("doc-file");'
            '  const title     = document.getElementById("doc-title").value.trim();'
            '  const result    = document.getElementById("doc-result");'
            '  if (!fileInput.files.length) { result.textContent="Select a file first."; result.className="submit-result err"; result.style.display="block"; return; }'
            '  result.textContent = "Ingesting… (this may take a minute)";'
            '  result.className = "submit-result ok"; result.style.display="block";'
            '  const fd = new FormData();'
            '  fd.append("file", fileInput.files[0]);'
            '  if (title) fd.append("title", title);'
            '  try {'
            '    const r = await fetch("/api/submit/doc", {method:"POST", body:fd});'
            '    const d = await r.json();'
            '    result.textContent = d.message || (r.ok ? "Done." : "Error.");'
            '    result.className = "submit-result " + (r.ok && d.ok !== false ? "ok" : "err");'
            '  } catch(e) { result.textContent="Network error: "+e; result.className="submit-result err"; }'
            '}'
            '</script>'
        )
        return _page("Submit", body, "submit")

    @app.get("/projects", response_class=HTMLResponse)
    async def page_projects() -> str:
        profile = _load(_BASE / "interest_profile.json", {})
        ideas   = profile.get("project_ideas", [])[::-1]  # newest first
        total   = len(ideas)

        _badge_labels = ["Idea", "Concept", "Build", "Research"]
        _badge_cls    = ["pb-0", "pb-1", "pb-2", "pb-3"]

        cards = ""
        for i, idea in enumerate(ideas[:120]):
            bi  = i % 4
            num = total - i
            cards += (
                f'<details class="project-card">'
                f'<summary>'
                f'<div class="proj-num">#{num}</div>'
                f'<span class="proj-badge {_badge_cls[bi]}">{_badge_labels[bi]}</span>'
                f'<div class="proj-text">{_escape(idea)}</div>'
                f'<div class="proj-expand">↓ expand</div>'
                f'</summary>'
                f'</details>'
            )

        if not cards:
            cards = '<div class="empty">No project ideas yet — ingest some content to get started.</div>'

        body = (
            f'<div class="page-header">'
            f'<div class="page-title">Projects</div>'
            f'<div class="page-sub">{total} idea{"s" if total != 1 else ""} extracted from ingested content</div>'
            f'</div>'
            f'<div class="project-grid">{cards}</div>'
        )
        return _page("Projects", body, "projects")

    @app.get("/profile", response_class=HTMLResponse)
    async def page_profile() -> str:
        profile = _load(_BASE / "interest_profile.json", {})
        topics  = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])[:40]
        people  = sorted(profile.get("people", {}).items(),  key=lambda x: -x[1])[:20]

        topics_html = "".join(f'<span class="tag">{_escape(t)} ({c})</span>' for t, c in topics)
        people_html = "".join(f'<span class="tag">{_escape(p)} ({c})</span>' for p, c in people)

        counts = (
            f'{profile.get("email_count", 0)} emails · '
            f'{profile.get("video_count", 0)} videos · '
            f'{profile.get("rss_count", 0)} articles · '
            f'{profile.get("podcast_count", 0)} podcasts'
        )

        body = (
            f'<div class="page-header">'
            f'<div class="page-title">Interest Profile</div>'
            f'<div class="page-sub">{counts}</div>'
            f'</div>'
            f'<h2>Topics</h2>'
            f'<div class="gap">{topics_html or "<div class=\'empty\'>None yet.</div>"}</div>'
            + (f'<h2>People & Orgs</h2><div class="gap">{people_html}</div>' if people_html else '')
            + f'<h2>Top Bars</h2><div class="gap">{_topics_html()}</div>'
        )
        return _page("Profile", body, "profile")

    @app.get("/wisdom", response_class=HTMLResponse)
    async def page_wisdom(offset: int = 0, limit: int = 50) -> str:
        store   = _load(_BASE / "wisdom_store.json", [])
        total   = len(store)
        entries = store[-(offset + limit):][::-1][:limit]

        icons = {"email": "✉", "youtube": "▶", "rss": "◈", "podcast": "◉"}
        rows  = ""
        for e in entries:
            src     = e.get("source", "")
            icon    = icons.get(src, "◌")
            title   = _escape(e.get("title", e.get("subject", ""))[:60])
            date    = (e.get("stored_at") or e.get("watched_at") or "")[:10]
            pats    = _escape(", ".join(e.get("patterns", [])))
            raw     = e.get("raw", {})
            preview = _escape((raw.get("summarize") or raw.get("extract_wisdom") or "")[:120])
            rows += (
                f'<tr>'
                f'<td style="width:24px;color:var(--text-dim)">{icon}</td>'
                f'<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{title}">{title or "—"}</td>'
                f'<td style="width:90px;color:var(--text-dim)">{date}</td>'
                f'<td style="width:110px;color:var(--text-dim)">{pats}</td>'
                f'<td style="color:var(--text-dim);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{preview}</td>'
                f'</tr>'
            )

        body = (
            f'<div class="page-header">'
            f'<div class="page-title">Wisdom Store</div>'
            f'<div class="page-sub">{total} entries</div>'
            f'</div>'
            f'<table class="tbl">'
            f'<thead><tr><th></th><th>Title</th><th>Date</th><th>Patterns</th><th>Preview</th></tr></thead>'
            f'<tbody>{rows or "<tr><td colspan=5 class=empty>No wisdom entries yet.</td></tr>"}</tbody>'
            f'</table>'
        )
        return _page("Wisdom", body, "wisdom")

    @app.get("/feeds", response_class=HTMLResponse)
    async def page_feeds() -> str:
        rss_feeds     = _load(_BASE / "rss_feeds.json", [])
        podcast_feeds = _load(_BASE / "podcast_feeds.json", [])

        def _feed_rows(feeds: list, kind: str) -> str:
            if not feeds:
                return '<tr><td colspan="2" class="empty" style="padding:20px">No feeds configured.</td></tr>'
            rows = ""
            for url in feeds:
                rows += (
                    f'<tr><td style="word-break:break-all">{_escape(url)}</td>'
                    f'<td style="width:90px">'
                    f'<form method="post" action="/htmx/{kind}/remove" style="display:inline">'
                    f'<input type="hidden" name="url" value="{_escape(url)}">'
                    f'<button type="submit" class="btn btn-danger btn-sm">remove</button>'
                    f'</form></td></tr>'
                )
            return rows

        body = (
            '<div class="page-header">'
            '<div class="page-title">Feeds</div>'
            '<div class="page-sub">Manage RSS and podcast sources</div>'
            '</div>'
            f'<h2>RSS ({len(rss_feeds)})</h2>'
            '<form class="form-block gap" style="padding:16px" method="post" action="/htmx/rss/add">'
            '<div class="form-row">'
            '<input type="url" name="url" placeholder="https://example.com/feed.rss" required>'
            '<button type="submit" class="btn">Add RSS Feed</button>'
            '</div></form>'
            f'<table class="tbl gap"><thead><tr><th>URL</th><th></th></tr></thead>'
            f'<tbody>{_feed_rows(rss_feeds, "rss")}</tbody></table>'

            f'<h2>Podcasts ({len(podcast_feeds)})</h2>'
            '<form class="form-block gap" style="padding:16px" method="post" action="/htmx/podcast/add">'
            '<div class="form-row">'
            '<input type="url" name="url" placeholder="https://example.com/podcast.rss" required>'
            '<button type="submit" class="btn">Add Podcast</button>'
            '</div></form>'
            f'<table class="tbl"><thead><tr><th>URL</th><th></th></tr></thead>'
            f'<tbody>{_feed_rows(podcast_feeds, "podcast")}</tbody></table>'
        )
        return _page("Feeds", body, "feeds")

    @app.get("/signals", response_class=HTMLResponse)
    async def page_signals() -> str:
        signals = _load(_BASE / "signals_log.json", [])[::-1][:50]
        cards   = ""
        for s in signals:
            src_tags = "".join(f'<span class="tag">{_escape(src)}</span>'
                               for src in s.get("sources", []))
            date = s.get("detected_at", "")[:10]
            cards += (
                f'<div class="signal-card">'
                f'<div class="signal-topic">◎ {_escape(s.get("topic", ""))}'
                f'<span style="color:var(--text-dim);font-size:12px;float:right">{date}</span></div>'
                f'<div class="signal-meta">{src_tags}'
                f' <span style="margin-left:8px">{s.get("hit_count", 0)} hits</span></div>'
                f'<div class="signal-snippet">{_escape(s.get("top_hit", "")[:200])}</div>'
                f'</div>'
            )

        lookback = os.getenv("SIGNAL_LOOKBACK_DAYS", "7")
        body = (
            f'<div class="page-header">'
            f'<div class="page-title">Signals</div>'
            f'<div class="page-sub">Concepts appearing across 2+ source types · {lookback}-day window</div>'
            f'</div>'
            + (cards or '<div class="empty">No signals detected yet. '
               'Signals appear after content from multiple sources overlaps.</div>')
        )
        return _page("Signals", body, "signals")

    @app.get("/feedback", response_class=HTMLResponse)
    async def page_feedback() -> str:
        fb   = _feedback_stats()
        rows = ""
        for entry in fb["recent"]:
            pos  = entry.get("positive")
            mark = '<span class="pos">+</span>' if pos else '<span class="neg">−</span>'
            ts   = entry.get("ts", "")[:16].replace("T", " ")
            q    = _escape(entry.get("user_query", "")[:80])
            r    = _escape(entry.get("bot_response_preview", "")[:100])
            rows += (
                f'<tr>'
                f'<td style="width:28px;font-size:16px;text-align:center">{mark}</td>'
                f'<td style="width:130px;color:var(--text-dim)">{ts}</td>'
                f'<td>{q or "—"}</td>'
                f'<td style="color:var(--text-dim)">{r or "—"}</td>'
                f'</tr>'
            )

        rate_str = f'{fb["rate"]}%' if fb["rate"] is not None else "—"
        body = (
            '<div class="page-header">'
            '<div class="page-title">Feedback</div>'
            '<div class="page-sub">Discord reaction history</div>'
            '</div>'
            '<div class="card-grid" style="max-width:500px">'
            f'<div class="card ok"><div class="card-label">Positive</div><div class="card-value">{fb["positive"]}</div></div>'
            f'<div class="card {"error" if fb["negative"] else ""}"><div class="card-label">Negative</div><div class="card-value">{fb["negative"]}</div></div>'
            f'<div class="card"><div class="card-label">Approval Rate</div><div class="card-value">{rate_str}</div></div>'
            '</div>'
            '<h2>Recent</h2>'
            '<table class="tbl">'
            '<thead><tr><th></th><th>Time</th><th>User query</th><th>Bot response</th></tr></thead>'
            f'<tbody>{rows or "<tr><td colspan=4 class=empty>No feedback yet.</td></tr>"}</tbody>'
            '</table>'
        )
        return _page("Feedback", body, "feedback")

    @app.get("/gcpdot", response_class=HTMLResponse)
    async def page_gcpdot() -> str:
        body = (
            '<div class="page-header">'
            '<div class="page-title">GCP Dot</div>'
            '<div class="page-sub">Global Consciousness Project — real-time coherence indicator</div>'
            '</div>'
            '<div style="display:flex;flex-direction:column;align-items:center;gap:28px;padding:40px 0">'

            # Dot iframe
            '<div style="'
            'background:var(--surface);border:1px solid var(--border);border-radius:16px;'
            'padding:36px 40px;display:flex;flex-direction:column;align-items:center;gap:20px;'
            'box-shadow:0 4px 32px rgba(0,0,0,.5);max-width:340px;width:100%'
            '">'
            '<iframe src="https://global-mind.org/gcpdot/gcp.html" '
            'height="160" width="160" scrolling="no" marginwidth="0" marginheight="0" '
            'frameborder="0" '
            'style="border:none;display:block;border-radius:8px;background:transparent" '
            'title="GCP Dot live indicator">'
            '</iframe>'

            # Legend
            '<div style="text-align:center;max-width:280px">'
            '<div style="font-size:13px;color:var(--text);font-weight:500;margin-bottom:8px">'
            'Global Consciousness Project</div>'
            '<div style="font-size:12px;color:var(--text-dim);line-height:1.6">'
            'Dot colour reflects statistical deviation across a worldwide network of '
            'random-number generators. Deep blue = significant coherence. '
            'Red/orange = notable deviation.'
            '</div>'
            '</div>'

            # Colour key
            '<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center">'
            + "".join(
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{bg};flex-shrink:0"></span>'
                f'<span style="font-size:11px;color:var(--text-dim)">{label}</span>'
                f'</div>'
                for bg, label in [
                    ("#1a66ff", "Significant"),
                    ("#00bbff", "Moderate"),
                    ("#aaaaaa", "Neutral"),
                    ("#ffaa00", "Notable"),
                    ("#ff2200", "Striking"),
                ]
            )
            + '</div>'
            '</div>'

            # Attribution link
            '<div style="font-size:11px;color:var(--text-dim);text-align:center">'
            'Data source: <a href="https://global-mind.org" target="_blank" rel="noopener">'
            'global-mind.org</a> &nbsp;·&nbsp; '
            '<a href="https://noosphere.princeton.edu" target="_blank" rel="noopener">'
            'noosphere.princeton.edu</a>'
            '</div>'
            '</div>'
        )
        return _page("GCP Dot", body, "gcpdot")

    @app.get("/chat", response_class=HTMLResponse)
    async def page_chat() -> str:
        body = (
            '<div class="page-header">'
            '<div class="page-title">Chat</div>'
            '<div class="page-sub">Talk to Buzlock directly from the browser</div>'
            '</div>'
            '<div style="display:flex;flex-direction:column;height:calc(100vh - 140px);gap:12px">'

            # Message wall
            '<div id="chat-wall" style="'
            'flex:1;background:var(--surface);border:1px solid var(--border);'
            'border-radius:10px;padding:16px;overflow-y:auto;'
            'font-size:13px;line-height:1.65;'
            '"></div>'

            # Input row
            '<div style="display:flex;gap:10px;flex-shrink:0">'
            '<textarea id="chat-input" rows="2" placeholder="Message Buzlock…" '
            'style="flex:1;resize:none;min-height:52px;padding:12px 14px;font-size:13px;'
            'background:var(--surface);border:1px solid var(--border);border-radius:8px;'
            'color:var(--text);font-family:inherit;outline:none;transition:border-color .15s" '
            'onkeydown="if((event.ctrlKey||event.metaKey)&&event.key===\'Enter\'){sendChat();event.preventDefault()}">'
            '</textarea>'
            '<button class="btn btn-teal" onclick="sendChat()" style="align-self:flex-end;height:52px;padding:0 22px">'
            'Send'
            '</button>'
            '</div>'
            '<div style="font-size:11px;color:var(--text-dim)">Ctrl+Enter to send</div>'
            '</div>'

            '<script>'
            'const chatWall = document.getElementById("chat-wall");'
            'function _escHtml(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}'
            'function _addMsg(role, text, cls) {'
            '  const d = document.createElement("div");'
            '  const color = role==="user" ? "var(--accent)" : role==="error" ? "var(--red)" : "var(--teal)";'
            '  const label = role==="user" ? "You" : role==="error" ? "Error" : "Buzlock";'
            '  d.innerHTML = `<div style="margin-bottom:14px">`'
            '    + `<span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:${color}">${label}</span>`'
            '    + `<div style="margin-top:4px;color:var(--text);white-space:pre-wrap">${_escHtml(text)}</div>`'
            '    + `</div>`;'
            '  chatWall.appendChild(d);'
            '  chatWall.scrollTop = chatWall.scrollHeight;'
            '}'
            'async function sendChat() {'
            '  const inp = document.getElementById("chat-input");'
            '  const msg = inp.value.trim(); if (!msg) return;'
            '  inp.value = ""; inp.disabled = true;'
            '  _addMsg("user", msg);'
            '  const thinking = document.createElement("div");'
            '  thinking.id = "chat-thinking";'
            '  thinking.innerHTML = `<span style="color:var(--text-dim);font-style:italic;font-size:12px">Buzlock is thinking…</span>`;'
            '  chatWall.appendChild(thinking);'
            '  chatWall.scrollTop = chatWall.scrollHeight;'
            '  try {'
            '    const r = await fetch("/api/chat", {method:"POST",'
            '      headers:{"Content-Type":"application/json"},'
            '      body:JSON.stringify({message:msg})});'
            '    const d = await r.json();'
            '    const t = document.getElementById("chat-thinking");'
            '    if (t) t.remove();'
            '    if (r.ok) _addMsg("bot", d.response || "…");'
            '    else _addMsg("error", d.detail || d.message || "Error");'
            '  } catch(e) {'
            '    const t = document.getElementById("chat-thinking"); if(t) t.remove();'
            '    _addMsg("error", "Network error: "+e);'
            '  }'
            '  inp.disabled = false; inp.focus();'
            '}'
            '</script>'
        )
        return _page("Chat", body, "chat")

    @app.get("/usage", response_class=HTMLResponse)
    async def page_usage() -> str:
        try:
            from core.usage_tracker import get_stats
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
        except Exception:
            s7 = s30 = {"total_calls":0,"total_tokens":0,"total_cost_usd":0.0,
                        "avg_latency_s":0.0,"cache_hit_rate":0,"by_model":{}}

        def _card(label, value, sub=""):
            return (f'<div class="card"><div class="card-label">{label}</div>'
                    f'<div class="card-value">{value}</div>'
                    f'{"<div class=card-sub>"+sub+"</div>" if sub else ""}</div>')

        rows7 = "".join(
            f'<tr><td>{m}</td><td>{d["calls"]}</td><td>{d["tokens"]:,}</td>'
            f'<td>${d["cost"]:.4f}</td><td>{d["latency"]/d["calls"]:.1f}s</td></tr>'
            for m, d in sorted(s7["by_model"].items(), key=lambda x: -x[1]["calls"])
        ) or '<tr><td colspan="5" class="empty">No data yet.</td></tr>'

        body = (
            '<div class="page-header"><div class="page-title">Usage</div>'
            '<div class="page-sub">LLM inference stats</div></div>'
            '<h2>Last 7 days</h2>'
            '<div class="card-grid">'
            + _card("Calls", f'{s7["total_calls"]:,}')
            + _card("Tokens", f'{s7["total_tokens"]:,}')
            + _card("Cost", f'${s7["total_cost_usd"]:.4f}')
            + _card("Avg Latency", f'{s7["avg_latency_s"]:.1f}s')
            + _card("Cache Hit Rate", f'{s7["cache_hit_rate"]}%', f'{s7["cache_hits"]} hits')
            + '</div>'
            '<h2>By model (7d)</h2>'
            '<table class="tbl"><thead><tr>'
            '<th>Model</th><th>Calls</th><th>Tokens</th><th>Cost</th><th>Avg Latency</th>'
            '</tr></thead><tbody>' + rows7 + '</tbody></table>'
            '<h2>Last 30 days</h2>'
            '<div class="card-grid">'
            + _card("Calls", f'{s30["total_calls"]:,}')
            + _card("Tokens", f'{s30["total_tokens"]:,}')
            + _card("Cost", f'${s30["total_cost_usd"]:.4f}')
            + '</div>'
        )
        return _page("Usage", body, "usage")

    @app.get("/bookmarklet", response_class=HTMLResponse)
    async def page_bookmarklet() -> str:
        import urllib.parse
        dashboard_url = os.getenv("DASHBOARD_PUBLIC_URL", "http://localhost:7860")
        bm_js = (
            "javascript:(function(){"
            "var u=encodeURIComponent(location.href);"
            "var t=encodeURIComponent(document.title);"
            f"window.open('{dashboard_url}/api/submit/url?url='+u+'&title='+t,'_blank','width=400,height=200');"
            "})();"
        )
        bm_encoded = _escape(bm_js)
        body = (
            '<div class="page-header"><div class="page-title">Bookmarklet</div>'
            '<div class="page-sub">Send any page to MYCONEX in one click</div></div>'
            '<div class="form-block" style="max-width:560px">'
            '<p style="font-size:13px;color:var(--text);margin-bottom:18px;line-height:1.6">'
            'Drag the button below to your browser\'s bookmarks bar. '
            'Click it on any page to instantly send that URL to your MYCONEX knowledge base.'
            '</p>'
            f'<a href="{bm_encoded}" '
            'style="display:inline-block;background:var(--teal-dim);color:var(--teal);'
            'border:1px solid var(--teal);border-radius:8px;padding:10px 22px;'
            'font-size:14px;font-weight:600;text-decoration:none;cursor:grab" '
            'onclick="alert(\'Drag this to your bookmarks bar — don\\\'t click it here\');return false">'
            '⊛ Add to MYCONEX'
            '</a>'
            '<div class="form-hint" style="margin-top:14px">'
            'Right-click → Bookmark this link if drag doesn\'t work in your browser.'
            '</div>'
            '<hr style="border:none;border-top:1px solid var(--border);margin:20px 0">'
            '<div style="font-size:11px;color:var(--text-dim)">'
            f'Submits to: <code style="color:var(--accent)">{dashboard_url}/api/submit/url</code><br>'
            'Set <code>DASHBOARD_PUBLIC_URL</code> in .env if your dashboard runs on a custom host/port.'
            '</div>'
            '</div>'
        )
        return _page("Bookmarklet", body, "bookmarklet")

    # ── SSE stream ────────────────────────────────────────────────────────────

    @app.get("/stream/activity")
    async def stream_activity(request: Request):
        import asyncio as _aio
        from core.notifications import subscribe, unsubscribe, get_recent

        q = subscribe()

        async def event_gen():
            try:
                # Send buffered history immediately
                for msg in get_recent(80):
                    payload = json.dumps({"text": msg})
                    yield f"data: {payload}\n\n"

                # Stream live
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await _aio.wait_for(q.get(), timeout=20.0)
                        yield f"data: {json.dumps({'text': msg})}\n\n"
                    except _aio.TimeoutError:
                        yield 'data: {"type":"ping"}\n\n'
            finally:
                unsubscribe(q)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/stream/log")
    async def stream_log(request: Request):
        import asyncio as _aio

        _LOG_FILE = Path("/tmp/buzlock.log")

        async def event_gen():
            # Send last 200 lines on connect
            try:
                if _LOG_FILE.exists():
                    lines = _LOG_FILE.read_text(errors="replace").splitlines()
                    for line in lines[-200:]:
                        if line.strip():
                            yield f"data: {json.dumps({'line': line})}\n\n"
            except Exception:
                pass

            # Tail: watch for new content
            pos = _LOG_FILE.stat().st_size if _LOG_FILE.exists() else 0
            while True:
                if await request.is_disconnected():
                    break
                await _aio.sleep(0.5)
                try:
                    if not _LOG_FILE.exists():
                        yield 'data: {"type":"ping"}\n\n'
                        continue
                    size = _LOG_FILE.stat().st_size
                    if size < pos:
                        pos = 0  # file was rotated/truncated
                    if size > pos:
                        with open(_LOG_FILE, errors="replace") as fh:
                            fh.seek(pos)
                            new = fh.read()
                        pos = size
                        for line in new.splitlines():
                            if line.strip():
                                yield f"data: {json.dumps({'line': line})}\n\n"
                except Exception:
                    yield 'data: {"type":"ping"}\n\n'

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Submit API ────────────────────────────────────────────────────────────

    @app.post("/api/submit/url")
    async def api_submit_url(request: Request) -> JSONResponse:
        import re
        import urllib.request as _ur
        from core.notifications import notify

        try:
            body = await request.json()
            url  = str(body.get("url", "")).strip()
            if not url or not url.startswith(("http://", "https://")):
                return JSONResponse({"ok": False, "message": "Invalid URL."}, status_code=400)

            req = _ur.Request(url, headers={"User-Agent": "MYCONEX/1.0"})
            with _ur.urlopen(req, timeout=20) as resp:
                raw = resp.read(400_000).decode("utf-8", errors="ignore")

            # Extract title + strip markup
            m_title = re.search(r"<title[^>]*>([^<]{1,120})</title>", raw, re.I)
            title   = m_title.group(1).strip() if m_title else url
            text    = re.sub(r"<[^>]+>", " ", raw)
            text    = re.sub(r"\s+", " ", text).strip()[:8000]

            entry = {
                "source":       "manual",
                "url":          url,
                "title":        title,
                "content":      text,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }

            stored = False
            try:
                from integrations.knowledge_store import embed_and_store
                eid = await embed_and_store(text, source="manual",
                                             metadata={"title": title, "url": url})
                stored = bool(eid)
            except Exception:
                pass

            if not stored:
                log_path = _BASE / "manual_submissions.json"
                entries  = _load(log_path, [])
                entries.append(entry)
                log_path.write_text(json.dumps(entries, indent=2))

            await notify(f"[submit] URL ingested: {title[:70]}")
            return JSONResponse({"ok": True, "message": f"Ingested: {title[:70]}"})

        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.post("/api/submit/text")
    async def api_submit_text(request: Request) -> JSONResponse:
        from core.notifications import notify

        try:
            body  = await request.json()
            text  = str(body.get("text", "")).strip()
            title = str(body.get("title", "Note")).strip() or "Note"
            if not text:
                return JSONResponse({"ok": False, "message": "Text is required."}, status_code=400)

            entry = {
                "source":       "manual",
                "title":        title,
                "content":      text,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }

            stored = False
            try:
                from integrations.knowledge_store import embed_and_store
                eid = await embed_and_store(text, source="manual",
                                             metadata={"title": title})
                stored = bool(eid)
            except Exception:
                pass

            if not stored:
                log_path = _BASE / "manual_submissions.json"
                entries  = _load(log_path, [])
                entries.append(entry)
                log_path.write_text(json.dumps(entries, indent=2))

            await notify(f"[submit] Note saved: {title[:70]}")
            return JSONResponse({"ok": True, "message": f"Saved: {title[:70]}"})

        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    # ── HTMX feed management ──────────────────────────────────────────────────

    @app.post("/htmx/rss/add", response_class=HTMLResponse)
    async def htmx_rss_add(url: str = Form(...)):
        try:
            from integrations.rss_monitor import RSSMonitor
            RSSMonitor().add_feed(url.strip())
        except Exception:
            pass
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/feeds", status_code=303)

    @app.post("/htmx/rss/remove", response_class=HTMLResponse)
    async def htmx_rss_remove(url: str = Form(...)):
        try:
            from integrations.rss_monitor import RSSMonitor
            RSSMonitor().remove_feed(url.strip())
        except Exception:
            pass
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/feeds", status_code=303)

    @app.post("/htmx/podcast/add", response_class=HTMLResponse)
    async def htmx_podcast_add(url: str = Form(...)):
        try:
            from integrations.podcast_ingester import PodcastIngester
            PodcastIngester().add_feed(url.strip())
        except Exception:
            pass
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/feeds", status_code=303)

    @app.post("/htmx/podcast/remove", response_class=HTMLResponse)
    async def htmx_podcast_remove(url: str = Form(...)):
        try:
            from integrations.podcast_ingester import PodcastIngester
            PodcastIngester().remove_feed(url.strip())
        except Exception:
            pass
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/feeds", status_code=303)

    # ── JSON API ──────────────────────────────────────────────────────────────

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse({"ingesters": _ingester_status()})

    @app.get("/api/profile")
    async def api_profile() -> JSONResponse:
        return JSONResponse(_load(_BASE / "interest_profile.json", {}))

    @app.get("/api/wisdom")
    async def api_wisdom(offset: int = 0, limit: int = 50) -> JSONResponse:
        store = _load(_BASE / "wisdom_store.json", [])
        return JSONResponse({
            "total":   len(store),
            "entries": store[-(offset + limit):][::-1][:limit],
        })

    @app.get("/api/signals")
    async def api_signals() -> JSONResponse:
        return JSONResponse(_load(_BASE / "signals_log.json", []))

    @app.get("/api/feedback")
    async def api_feedback() -> JSONResponse:
        return JSONResponse(_feedback_stats())

    @app.get("/api/sysmon")
    async def api_sysmon() -> JSONResponse:
        return JSONResponse(_sysmon_data())

    @app.post("/api/chat")
    async def api_chat(request: Request) -> JSONResponse:
        """Direct chat endpoint — calls the same Ollama pipeline as Discord."""
        try:
            body = await request.json()
            message = str(body.get("message", "")).strip()
            if not message:
                return JSONResponse({"ok": False, "message": "Empty message"}, status_code=400)

            import httpx
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model      = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

            # Try RAG context
            rag_ctx = ""
            try:
                from integrations.knowledge_store import search
                hits = await search(message, limit=3)
                if hits:
                    rag_ctx = "\n\n[Knowledge base context]\n" + "\n---\n".join(
                        h.get("text", "")[:300] for h in hits
                    )
            except Exception:
                pass

            system = (
                "You are Buzlock, a personal AI assistant. "
                "Answer concisely and helpfully."
                + rag_ctx
            )

            t0 = time.time()
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": message},
                        ],
                    },
                )
            latency = time.time() - t0
            resp_text = r.json().get("message", {}).get("content", "").strip()

            try:
                from core.usage_tracker import record as _ur
                _ur(model=model, prompt=message, completion=resp_text,
                    latency_s=latency, source="dashboard")
            except Exception:
                pass

            return JSONResponse({"ok": True, "response": resp_text or "…"})
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.post("/api/ingest/webhook")
    async def api_webhook(request: Request) -> JSONResponse:
        """
        Generic webhook ingestion endpoint.
        Accepts JSON with any of: url, text, title, source, tags.
        Protected by WEBHOOK_SECRET env var if set.
        """
        secret = os.getenv("WEBHOOK_SECRET", "")
        if secret:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {secret}" and auth != secret:
                return JSONResponse({"ok": False, "message": "Unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "message": "Invalid JSON"}, status_code=400)

        url   = body.get("url", "").strip()
        text  = body.get("text", "").strip()
        title = body.get("title", "untitled")

        if url:
            # Re-use the existing URL submit pipeline
            try:
                from integrations.knowledge_store import store_item
                await store_item(text=f"{title}\n{url}", source=body.get("source","webhook"),
                                 title=title)
                return JSONResponse({"ok": True, "message": f"URL queued: {url}"})
            except Exception as exc:
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)
        elif text:
            try:
                from integrations.knowledge_store import store_item
                item_id = await store_item(text=text, source=body.get("source","webhook"),
                                           title=title)
                return JSONResponse({"ok": True, "message": f"Stored {item_id or 'item'}"})
            except Exception as exc:
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)
        else:
            return JSONResponse({"ok": False, "message": "Provide 'url' or 'text'"}, status_code=400)

    @app.get("/api/usage")
    async def api_usage(days: int = 7) -> JSONResponse:
        try:
            from core.usage_tracker import get_stats
            return JSONResponse(get_stats(days=days))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/submit/doc")
    async def api_submit_doc(file: UploadFile = File(...), title: str = Form("")) -> JSONResponse:
        try:
            from integrations.document_ingester import ingest_document
            data  = await file.read()
            fname = file.filename or "upload.txt"
            result = await ingest_document(data, fname, title=title or "")
            if result.get("ok"):
                return JSONResponse({"ok": True, "message":
                    f"Ingested '{result['title']}' — {result['chunks']} chunk(s), "
                    f"{len(result.get('topics', []))} topic(s)"})
            else:
                return JSONResponse({"ok": False, "message": result.get("error", "Unknown error")}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        """Prometheus-compatible text metrics."""
        lines: list[str] = []

        def g(name: str, value: float, help_text: str = "", labels: str = "") -> None:
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
            tag = "{" + labels + "}" if labels else ""
            lines.append(f"{name}{tag} {value}")

        # Ingester item counts
        for src, fname in [("email","email_insights.json"),("youtube","youtube_insights.json"),
                           ("rss","rss_insights.json"),("podcast","podcast_insights.json"),
                           ("github","github_insights.json"),("document","doc_insights.json")]:
            data = _load(_BASE / fname, [])
            g("myconex_ingested_total", len(data),
              f"Total items ingested from {src}", f'source="{src}"')

        # Feedback
        fb = _feedback_stats()
        g("myconex_feedback_positive", fb["positive"], "Positive feedback reactions")
        g("myconex_feedback_negative", fb["negative"], "Negative feedback reactions")

        # RAG gaps
        try:
            from core.rag_repair import get_open_gaps
            g("myconex_rag_gaps_open", len(get_open_gaps()), "Open RAG knowledge gaps")
        except Exception:
            pass

        # Signals
        sigs = _load(_BASE / "signals_log.json", [])
        g("myconex_signals_total", len(sigs), "Total cross-source signals detected")

        # System
        if _PSUTIL_OK:
            import psutil
            g("myconex_cpu_percent", psutil.cpu_percent(interval=0.1), "CPU usage percent")
            g("myconex_mem_percent", psutil.virtual_memory().percent, "Memory usage percent")
            g("myconex_disk_percent", psutil.disk_usage("/").percent, "Disk usage percent")

        return "\n".join(lines) + "\n"

    @app.get("/manifest.json")
    async def pwa_manifest() -> JSONResponse:
        return JSONResponse({
            "name": "MYCONEX",
            "short_name": "MYCONEX",
            "description": "Personal AI knowledge mesh dashboard",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#07070f",
            "theme_color": "#07070f",
            "icons": [
                {"src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⊛</text></svg>",
                 "sizes": "any", "type": "image/svg+xml"}
            ],
            "categories": ["productivity", "utilities"],
        })

    @app.get("/sw.js", response_class=PlainTextResponse)
    async def service_worker() -> str:
        return """
const CACHE = 'myconex-v1';
const OFFLINE_URLS = ['/'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(OFFLINE_URLS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
"""

else:
    app = None  # type: ignore[assignment]


# ── Standalone runner ──────────────────────────────────────────────────────────

async def start_dashboard(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the dashboard server as an asyncio background task."""
    if not _FASTAPI_OK:
        logger.warning("[dashboard] FastAPI not installed — run: pip install fastapi uvicorn")
        return
    try:
        import uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        logger.info("[dashboard] starting at http://%s:%d", host, port)
        await server.serve()
    except ImportError:
        logger.warning("[dashboard] uvicorn not installed — run: pip install uvicorn")
    except SystemExit:
        logger.warning("[dashboard] port %d already in use — dashboard not started", port)
    except Exception as exc:
        logger.warning("[dashboard] server error: %s", exc)
