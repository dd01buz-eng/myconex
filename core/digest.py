"""
MYCONEX Weekly Digest
----------------------
Builds a rich Discord embed (or plain-text fallback) summarising everything
Buzlock learned in the past week: top topics, new project ideas, source counts,
standout wisdom quotes, and feedback stats.

Data sources (all local JSON files):
  ~/.myconex/interest_profile.json  — accumulated interest profile
  ~/.myconex/wisdom_store.json      — Fabric-extracted wisdom entries
  ~/.myconex/feedback_log.jsonl     — 👍/👎 reactions log
  ~/.myconex/email_insights.json    — email processing log
  ~/.myconex/youtube_insights.json  — YouTube processing log
  ~/.myconex/rss_insights.json      — RSS processing log
  ~/.myconex/podcast_insights.json  — podcast processing log

Usage:
    from core.digest import build_digest_embed, schedule_weekly_digest
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE = Path.home() / ".myconex"
_PROFILE_FILE   = _BASE / "interest_profile.json"
_WISDOM_FILE    = _BASE / "wisdom_store.json"
_FEEDBACK_FILE  = _BASE / "feedback_log.jsonl"
_EMAIL_FILE     = _BASE / "email_insights.json"
_YT_FILE        = _BASE / "youtube_insights.json"
_RSS_FILE       = _BASE / "rss_insights.json"
_PODCAST_FILE   = _BASE / "podcast_insights.json"
_DIGEST_STAMP   = _BASE / "last_digest.txt"  # ISO timestamp of last sent digest
_SIGNALS_FILE   = _BASE / "signals_log.json"

# Post on Sunday at this UTC hour (inclusive window: hour N to N+1)
DIGEST_DAY  = int(os.getenv("DIGEST_DAY",  "6"))   # 0=Mon … 6=Sun
DIGEST_HOUR = int(os.getenv("DIGEST_HOUR", "9"))    # 9am UTC


# ── JSON helpers ──────────────────────────────────────────────────────────────

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


# ── Digest data assembly ──────────────────────────────────────────────────────

def _recent_since(entries: list[dict], days: int = 7) -> list[dict]:
    """Filter entries processed within the last N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = []
    for e in entries:
        ts = e.get("processed_at") or e.get("stored_at") or e.get("ts", "")
        if ts >= cutoff:
            result.append(e)
    return result


def build_digest_data(days: int = 7) -> dict[str, Any]:
    """Assemble all digest data into a single dict."""
    profile  = _load(_PROFILE_FILE, {})
    wisdom   = _load(_WISDOM_FILE, [])
    feedback = _load_jsonl(_FEEDBACK_FILE)

    # Count recent ingester activity
    email_recent   = _recent_since(_load(_EMAIL_FILE, []),   days)
    yt_recent      = _recent_since(_load(_YT_FILE, []),      days)
    rss_recent     = _recent_since(_load(_RSS_FILE, []),     days)
    podcast_recent = _recent_since(_load(_PODCAST_FILE, []), days)

    # Top topics (frequency-sorted)
    topics = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])
    top_topics = [t for t, _ in topics[:8]]

    # New project ideas added recently — use last N from list
    ideas = profile.get("project_ideas", [])[-10:]

    # Standout wisdom: pick the newest 3 entries that have extract_wisdom
    wisdom_recent = _recent_since(wisdom, days)
    quotes = []
    for entry in reversed(wisdom_recent):
        raw = entry.get("raw", {})
        text = raw.get("extract_wisdom") or raw.get("summarize") or ""
        if not text:
            continue
        # Grab the first bullet or sentence
        for line in text.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if len(line) > 40:
                src = entry.get("feed_title") or entry.get("title") or entry.get("source", "")
                quotes.append((line[:200], src))
                break
        if len(quotes) >= 3:
            break

    # Cross-source signals detected within the window
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    all_signals = _load(_SIGNALS_FILE, [])
    recent_signals = [
        s for s in all_signals
        if s.get("detected_at", "") >= cutoff
    ]
    # De-dup by topic, keep most recent per topic
    seen_topics: dict[str, dict] = {}
    for s in recent_signals:
        t = s.get("topic", "")
        if t not in seen_topics or s["detected_at"] > seen_topics[t]["detected_at"]:
            seen_topics[t] = s
    signals = sorted(seen_topics.values(), key=lambda s: s["detected_at"], reverse=True)[:8]

    # Feedback stats
    fb_total = len(feedback)
    fb_pos   = sum(1 for f in feedback if f.get("positive"))
    fb_neg   = fb_total - fb_pos
    fb_rate  = round(fb_pos / fb_total * 100) if fb_total else None

    # Total knowledge base counts from profile
    total_email   = profile.get("email_count", 0)
    total_video   = profile.get("video_count", 0)
    total_rss     = profile.get("rss_count", 0)
    total_podcast = profile.get("podcast_count", 0)

    return {
        "days":           days,
        "top_topics":     top_topics,
        "ideas":          ideas,
        "quotes":         quotes,
        "signals":        signals,
        "email_recent":   len(email_recent),
        "yt_recent":      len(yt_recent),
        "rss_recent":     len(rss_recent),
        "podcast_recent": len(podcast_recent),
        "total_email":    total_email,
        "total_video":    total_video,
        "total_rss":      total_rss,
        "total_podcast":  total_podcast,
        "fb_total":       fb_total,
        "fb_pos":         fb_pos,
        "fb_neg":         fb_neg,
        "fb_rate":        fb_rate,
        "last_updated":   profile.get("last_updated", ""),
    }


def build_digest_text(days: int = 7) -> str:
    """Plain-text digest (for non-discord uses or fallback)."""
    d = build_digest_data(days)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"📊 MYCONEX Weekly Digest  [{now}]",
        "=" * 48,
        "",
        f"📥 Ingested this week:",
        f"   📧 {d['email_recent']} emails   "
        f"📺 {d['yt_recent']} videos   "
        f"📰 {d['rss_recent']} articles   "
        f"🎙️ {d['podcast_recent']} podcast episodes",
        "",
    ]
    if d["top_topics"]:
        lines += ["🏷️ Top topics:", "   " + ", ".join(d["top_topics"]), ""]
    if d["ideas"]:
        lines += ["💡 Recent project ideas:"]
        for idea in d["ideas"][-5:]:
            lines.append(f"   • {idea[:120]}")
        lines.append("")
    if d["quotes"]:
        lines += ["✨ Standout wisdom:"]
        for quote, src in d["quotes"]:
            src_str = f"  — {src}" if src else ""
            lines.append(f'   "{quote}"{src_str}')
        lines.append("")
    if d["signals"]:
        lines += ["🔔 Cross-source signals:"]
        for s in d["signals"]:
            src_str = " + ".join(s.get("sources", []))
            lines.append(f"   • {s['topic']}  [{src_str}]")
        lines.append("")
    if d["fb_total"]:
        rate_str = f" ({d['fb_rate']}% positive)" if d["fb_rate"] is not None else ""
        lines.append(f"👍 Feedback: {d['fb_pos']} up / {d['fb_neg']} down{rate_str}")
    lines += [
        "",
        f"📚 All-time: {d['total_email']} emails · {d['total_video']} videos · "
        f"{d['total_rss']} articles · {d['total_podcast']} podcasts",
    ]
    return "\n".join(lines)


def build_digest_embed(days: int = 7) -> dict[str, Any]:
    """
    Build a Discord embed dict for the weekly digest.
    Returns a dict that can be passed to discord.Embed(**build_digest_embed()).
    """
    d = build_digest_data(days)
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")

    fields = []

    # Ingestion counts
    ingested_val = (
        f"📧 **{d['email_recent']}** emails\n"
        f"📺 **{d['yt_recent']}** videos\n"
        f"📰 **{d['rss_recent']}** articles\n"
        f"🎙️ **{d['podcast_recent']}** podcast episodes"
    )
    fields.append({"name": f"📥 Ingested (last {days}d)", "value": ingested_val, "inline": True})

    # All-time totals
    totals_val = (
        f"📧 {d['total_email']} emails\n"
        f"📺 {d['total_video']} videos\n"
        f"📰 {d['total_rss']} articles\n"
        f"🎙️ {d['total_podcast']} podcasts"
    )
    fields.append({"name": "📚 All-time", "value": totals_val, "inline": True})

    # Feedback
    if d["fb_total"]:
        rate_str = f"\n{d['fb_rate']}% positive" if d["fb_rate"] is not None else ""
        fb_val = f"👍 {d['fb_pos']}  👎 {d['fb_neg']}{rate_str}"
        fields.append({"name": "🗳️ Feedback", "value": fb_val, "inline": True})

    # Top topics
    if d["top_topics"]:
        fields.append({
            "name": "🏷️ Top topics",
            "value": "  ".join(f"`{t}`" for t in d["top_topics"][:6]),
            "inline": False,
        })

    # Project ideas
    if d["ideas"]:
        idea_lines = "\n".join(f"• {i[:100]}" for i in d["ideas"][-5:])
        fields.append({"name": "💡 Project ideas", "value": idea_lines, "inline": False})

    # Standout wisdom
    if d["quotes"]:
        q_lines = []
        for quote, src in d["quotes"]:
            src_str = f" *— {src[:40]}*" if src else ""
            q_lines.append(f'"{quote[:180]}"{src_str}')
        fields.append({
            "name": "✨ Standout wisdom",
            "value": "\n\n".join(q_lines),
            "inline": False,
        })

    # Cross-source signals
    if d["signals"]:
        _EMOJI = {"email": "📧", "youtube": "📺", "rss": "📰", "podcast": "🎙️"}
        sig_lines = []
        for s in d["signals"]:
            src_icons = " + ".join(_EMOJI.get(src, "📌") + src for src in s.get("sources", []))
            sig_lines.append(f"**{s['topic']}** — {src_icons}")
        fields.append({
            "name": "🔔 Cross-source signals",
            "value": "\n".join(sig_lines),
            "inline": False,
        })

    return {
        "title":       f"📊 MYCONEX Weekly Digest — {now}",
        "color":       0x00CFFF,
        "description": f"Everything Buzlock learned in the past {days} days.",
        "fields":      fields,
        "footer":      {"text": "Use /digest anytime to generate a fresh digest."},
    }


# ── Scheduler ─────────────────────────────────────────────────────────────────

def _digest_due() -> bool:
    """Return True if a weekly digest should be sent right now."""
    now = datetime.now(timezone.utc)
    if now.weekday() != DIGEST_DAY or now.hour != DIGEST_HOUR:
        return False
    # Check stamp — avoid re-sending if we already sent one this week
    try:
        if _DIGEST_STAMP.exists():
            last = datetime.fromisoformat(_DIGEST_STAMP.read_text().strip())
            if (now - last).days < 6:
                return False
    except Exception:
        pass
    return True


def _mark_digest_sent() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    _DIGEST_STAMP.write_text(datetime.now(timezone.utc).isoformat())


async def schedule_weekly_digest(post_fn) -> None:
    """
    Background coroutine that calls post_fn(embed_dict) when a digest is due.

    post_fn is an async callable that accepts the embed dict and posts it to
    the Discord home channel.  Runs forever (cancelled on shutdown).

    Example wiring in discord_gateway.py:
        asyncio.create_task(schedule_weekly_digest(self._post_digest))
    """
    logger.info("[digest] scheduler started — day=%d hour=%d UTC", DIGEST_DAY, DIGEST_HOUR)
    while True:
        await asyncio.sleep(60)   # check every minute
        if _digest_due():
            try:
                logger.info("[digest] sending weekly digest")
                embed_data = build_digest_embed()
                await post_fn(embed_data)
                _mark_digest_sent()
            except Exception as exc:
                logger.warning("[digest] error sending digest: %s", exc)
