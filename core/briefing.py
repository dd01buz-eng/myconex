"""
MYCONEX Morning Briefing
--------------------------
Proactive daily briefing pushed to Discord without any user command.

Runs at BRIEFING_HOUR (default 8am local time) every day.  Builds a compact
summary of what happened in the last 24h:
  - New emails processed and top topics extracted
  - RSS / YouTube / podcast items ingested
  - Cross-source signals detected overnight
  - Upcoming calendar events (if Google Calendar is wired up)
  - Any new project ideas extracted from content
  - Inference cache hit rate for the day (light system health)

Env vars:
  BRIEFING_HOUR          — local 24h hour to send (default: 8)
  BRIEFING_TIMEZONE      — tz name for scheduling, e.g. "America/New_York" (default: local)
  BRIEFING_ENABLED       — "false" to disable (default: true)
  BRIEFING_LOOKBACK_H    — hours to look back for "overnight" activity (default: 16)

Usage (wired in buzlock_bot.py):
    from core.briefing import MorningBriefing
    briefing = MorningBriefing(post_fn)
    asyncio.create_task(briefing.run_forever())
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

_BASE           = Path.home() / ".myconex"
_PROFILE_FILE   = _BASE / "interest_profile.json"
_EMAIL_FILE     = _BASE / "email_insights.json"
_YT_FILE        = _BASE / "youtube_insights.json"
_RSS_FILE       = _BASE / "rss_insights.json"
_PODCAST_FILE   = _BASE / "podcast_insights.json"
_SIGNALS_FILE   = _BASE / "signals_log.json"
_WISDOM_FILE    = _BASE / "wisdom_store.json"
_BRIEFING_STAMP = _BASE / "last_briefing.txt"

BRIEFING_HOUR     = int(os.getenv("BRIEFING_HOUR",     "8"))
BRIEFING_ENABLED  = os.getenv("BRIEFING_ENABLED", "true").lower() != "false"
BRIEFING_LOOKBACK = int(os.getenv("BRIEFING_LOOKBACK_H", "16"))  # hours back


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _since(entries: list[dict], hours: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    out = []
    for e in entries:
        ts = e.get("processed_at") or e.get("stored_at") or e.get("detected_at") or e.get("ts", "")
        if ts >= cutoff:
            out.append(e)
    return out


# ── Build briefing data ────────────────────────────────────────────────────────

def build_briefing_data(lookback_hours: int = BRIEFING_LOOKBACK) -> dict[str, Any]:
    profile  = _load(_PROFILE_FILE, {})
    emails   = _since(_load(_EMAIL_FILE,   []), lookback_hours)
    yt       = _since(_load(_YT_FILE,      []), lookback_hours)
    rss      = _since(_load(_RSS_FILE,     []), lookback_hours)
    podcasts = _since(_load(_PODCAST_FILE, []), lookback_hours)
    signals  = _since(_load(_SIGNALS_FILE, []), lookback_hours)
    wisdom   = _since(_load(_WISDOM_FILE,  []), lookback_hours)

    # Top topics across all recent activity
    topic_freq: dict[str, int] = {}
    for e in emails + rss + yt + podcasts:
        for t in e.get("topics", []):
            topic_freq[t] = topic_freq.get(t, 0) + 1
    top_topics = [t for t, _ in sorted(topic_freq.items(), key=lambda x: -x[1])[:5]]

    # New project ideas from recent content
    ideas: list[str] = []
    for e in emails + rss + yt:
        ideas.extend(e.get("project_ideas", []))
    ideas = list(dict.fromkeys(ideas))[:5]  # dedup, keep order

    # Standout wisdom quote (newest one)
    quote = ""
    quote_src = ""
    for entry in reversed(wisdom):
        raw = entry.get("raw", {})
        text = raw.get("extract_wisdom") or raw.get("summarize") or ""
        for line in text.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if len(line) > 40:
                quote = line[:200]
                quote_src = entry.get("feed_title") or entry.get("title") or entry.get("source", "")
                break
        if quote:
            break

    return {
        "lookback_hours": lookback_hours,
        "email_count":   len(emails),
        "yt_count":      len(yt),
        "rss_count":     len(rss),
        "podcast_count": len(podcasts),
        "signal_count":  len(signals),
        "signals":       signals[:4],
        "top_topics":    top_topics,
        "ideas":         ideas,
        "quote":         quote,
        "quote_src":     quote_src,
        "total_ingested": len(emails) + len(yt) + len(rss) + len(podcasts),
    }


async def build_briefing_embed(lookback_hours: int = BRIEFING_LOOKBACK) -> dict[str, Any]:
    """Build a Discord embed dict for the morning briefing."""
    d = build_briefing_data(lookback_hours)
    now = datetime.now().strftime("%A, %B %d")  # e.g. "Monday, March 25"
    fields = []

    # Activity summary
    if d["total_ingested"]:
        parts = []
        if d["email_count"]:   parts.append(f"📧 **{d['email_count']}** emails")
        if d["rss_count"]:     parts.append(f"📰 **{d['rss_count']}** articles")
        if d["yt_count"]:      parts.append(f"📺 **{d['yt_count']}** videos")
        if d["podcast_count"]: parts.append(f"🎙️ **{d['podcast_count']}** episodes")
        fields.append({
            "name": f"📥 Overnight ({lookback_hours}h)",
            "value": "\n".join(parts) or "Nothing new",
            "inline": True,
        })

    # Signals
    if d["signals"]:
        _EMOJI = {"email": "📧", "youtube": "📺", "rss": "📰", "podcast": "🎙️"}
        sig_lines = []
        for s in d["signals"]:
            icons = " + ".join(_EMOJI.get(src, "📌") + src for src in s.get("sources", [])[:2])
            sig_lines.append(f"**{s['topic']}** — {icons}")
        fields.append({
            "name": "🔔 Signals",
            "value": "\n".join(sig_lines),
            "inline": True,
        })

    # Top topics
    if d["top_topics"]:
        fields.append({
            "name": "🏷️ Trending",
            "value": "  ".join(f"`{t}`" for t in d["top_topics"]),
            "inline": False,
        })

    # Project ideas
    if d["ideas"]:
        fields.append({
            "name": "💡 New ideas",
            "value": "\n".join(f"• {i[:100]}" for i in d["ideas"]),
            "inline": False,
        })

    # Standout wisdom
    if d["quote"]:
        src_str = f" *— {d['quote_src'][:40]}*" if d["quote_src"] else ""
        fields.append({
            "name": "✨ Overnight wisdom",
            "value": f'"{d["quote"]}"{src_str}',
            "inline": False,
        })

    # Calendar events
    try:
        from integrations.calendar_ingester import get_upcoming_events, format_event_time
        events = await get_upcoming_events(lookahead_hours=12)
        if events:
            ev_lines = []
            for e in events[:5]:
                t = format_event_time(e)
                meet = " 🎥" if e.get("meet_link") else ""
                ev_lines.append(f"• {t}  **{e['summary']}**{meet}")
            fields.append({
                "name": "📅 Today's schedule",
                "value": "\n".join(ev_lines),
                "inline": False,
            })
    except Exception:
        pass

    # RAG gaps summary
    try:
        from core.rag_repair import get_open_gaps
        open_gaps = get_open_gaps(max_n=5)
        if open_gaps:
            gap_lines = [f"• {g['query'][:80]}" for g in open_gaps[:4]]
            if len(open_gaps) > 4:
                gap_lines.append(f"  _...and {len(open_gaps)-4} more. Use /gaps_")
            fields.append({
                "name": "⚠️ Knowledge gaps",
                "value": "\n".join(gap_lines),
                "inline": False,
            })
    except Exception:
        pass

    if not fields:
        return {}  # nothing to report — skip sending

    return {
        "title":       f"☀️ Good morning — {now}",
        "color":       0xFFB347,
        "description": (
            f"Here's what happened while you were away "
            f"({d['total_ingested']} item(s) processed)."
            if d["total_ingested"] else
            "Quiet night — nothing new to report."
        ),
        "fields": fields,
        "footer": {"text": f"MYCONEX · {lookback_hours}h lookback · /digest for full weekly summary"},
    }


# ── Scheduler ─────────────────────────────────────────────────────────────────

def _briefing_due() -> bool:
    """Return True if we should send the morning briefing right now."""
    now = datetime.now()
    if now.hour != BRIEFING_HOUR:
        return False
    try:
        if _BRIEFING_STAMP.exists():
            last_str = _BRIEFING_STAMP.read_text().strip()
            last = datetime.fromisoformat(last_str)
            if (now - last).total_seconds() < 23 * 3600:
                return False  # already sent today
    except Exception:
        pass
    return True


def _mark_sent() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    _BRIEFING_STAMP.write_text(datetime.now().isoformat())


class MorningBriefing:
    """
    Background task that posts a daily morning briefing embed to Discord.

    post_fn: async callable that accepts the embed dict and posts it.
    """

    def __init__(self, post_fn: Callable[[dict], Coroutine]) -> None:
        self._post_fn = post_fn
        self._running = False

    async def run_forever(self) -> None:
        if not BRIEFING_ENABLED:
            logger.info("[briefing] disabled via BRIEFING_ENABLED=false")
            return
        self._running = True
        logger.info("[briefing] scheduler started — fires daily at %02d:00 local", BRIEFING_HOUR)
        while self._running:
            try:
                if _briefing_due():
                    embed = await build_briefing_embed()
                    if embed:
                        await self._post_fn(embed)
                        _mark_sent()
                        logger.info("[briefing] morning briefing posted")
                    else:
                        _mark_sent()  # quiet night — mark anyway to avoid retrying
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[briefing] error: %s", exc)
            await asyncio.sleep(60)  # check every minute

    def stop(self) -> None:
        self._running = False
