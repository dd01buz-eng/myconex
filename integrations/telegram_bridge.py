"""
MYCONEX NATS-to-Telegram Bridge
---------------------------------
Thin bridge so the mesh can reach the user outside Discord.

Subscribes to:
  mesh.broadcast          — general mesh announcements
  mesh.notify.telegram    — targeted Telegram notifications
  mesh.digest.post        — digest posts (same hook as Discord home channel)

Also forwards any critical log messages flagged via `notify_telegram()`.

Env vars:
  TELEGRAM_BOT_TOKEN      — required; BotFather token
  TELEGRAM_CHAT_ID        — required; numeric chat/group/channel ID (can be negative for groups)
  TELEGRAM_NATS_URL       — optional; defaults to NATS_URL env var
  TELEGRAM_MAX_MSG_LEN    — max chars per message (default: 4096)
  TELEGRAM_PARSE_MODE     — "HTML" or "Markdown" (default: HTML)
  TELEGRAM_ENABLED        — "false" to disable (default: true)

Usage (wire into buzlock_bot.py):
    from integrations.telegram_bridge import TelegramBridge
    bridge = TelegramBridge()
    asyncio.create_task(bridge.run_forever())
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
_NATS_URL   = os.getenv("TELEGRAM_NATS_URL") or os.getenv("NATS_URL", "")
_MAX_LEN    = int(os.getenv("TELEGRAM_MAX_MSG_LEN", "4096"))
_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
_ENABLED    = os.getenv("TELEGRAM_ENABLED", "true").lower() != "false"


# ── Low-level Telegram API ────────────────────────────────────────────────────

async def _send_message(text: str, parse_mode: str = _PARSE_MODE) -> bool:
    """Send *text* to the configured Telegram chat. Returns True on success."""
    if not _TOKEN or not _CHAT_ID:
        logger.warning("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return False

    import urllib.request, urllib.error, urllib.parse

    text = text[:_MAX_LEN]
    payload = json.dumps({
        "chat_id":    _CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }).encode()

    url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                logger.warning("[telegram] sendMessage failed: %s", result)
                return False
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        logger.warning("[telegram] HTTP %d: %s", exc.code, body[:200])
        return False
    except Exception as exc:
        logger.warning("[telegram] send error: %s", exc)
        return False


async def notify_telegram(text: str) -> bool:
    """Public helper — send a plain notification to Telegram."""
    if not _ENABLED:
        return False
    return await _send_message(text)


# ── Embed → Telegram text conversion ─────────────────────────────────────────

def _embed_to_text(embed: dict[str, Any]) -> str:
    """Convert a Discord-style embed dict to a readable Telegram HTML message."""
    parts: list[str] = []

    title = embed.get("title", "")
    if title:
        parts.append(f"<b>{_esc(title)}</b>")

    desc = embed.get("description", "")
    if desc:
        parts.append(_esc(desc))

    for field in embed.get("fields", []):
        name  = field.get("name", "")
        value = field.get("value", "")
        if name or value:
            parts.append(f"\n<b>{_esc(name)}</b>\n{_esc(value)}")

    footer = embed.get("footer", {})
    if isinstance(footer, dict) and footer.get("text"):
        parts.append(f"\n<i>{_esc(footer['text'])}</i>")

    return "\n".join(parts)


def _esc(s: str) -> str:
    """Minimal HTML escape for Telegram HTML parse mode."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── NATS subscriber ───────────────────────────────────────────────────────────

class TelegramBridge:
    """
    Subscribes to NATS mesh subjects and forwards relevant messages to Telegram.
    Falls back to standalone notify mode if NATS is unavailable.
    """

    _SUBJECTS = [
        "mesh.broadcast",
        "mesh.notify.telegram",
        "mesh.digest.post",
    ]

    def __init__(self) -> None:
        self._nc: Optional[Any] = None
        self._subs: list[Any] = []
        self._running = False
        self._last_send: float = 0.0
        self._rate_limit = 1.0  # min seconds between messages

    async def run_forever(self) -> None:
        if not _ENABLED:
            logger.info("[telegram] bridge disabled via TELEGRAM_ENABLED=false")
            return
        if not _TOKEN or not _CHAT_ID:
            logger.warning("[telegram] bridge inactive — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
            return

        self._running = True
        logger.info("[telegram] bridge starting — chat_id=%s", _CHAT_ID)

        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[telegram] connection error: %s — retry in 30s", exc)
                await asyncio.sleep(30)

    async def _connect_and_listen(self) -> None:
        if not _NATS_URL:
            # No NATS — run in polling-only mode (just keep alive for direct calls)
            logger.info("[telegram] no NATS_URL — bridge in direct-notify mode only")
            while self._running:
                await asyncio.sleep(60)
            return

        try:
            import nats
        except ImportError:
            logger.warning("[telegram] nats-py not installed — bridge in direct-notify mode")
            while self._running:
                await asyncio.sleep(60)
            return

        self._nc = await nats.connect(_NATS_URL)
        logger.info("[telegram] connected to NATS %s", _NATS_URL)

        for subject in self._SUBJECTS:
            sub = await self._nc.subscribe(subject, cb=self._on_nats_message)
            self._subs.append(sub)
            logger.debug("[telegram] subscribed to %s", subject)

        try:
            while self._running:
                await asyncio.sleep(5)
        finally:
            for sub in self._subs:
                try:
                    await sub.unsubscribe()
                except Exception:
                    pass
            self._subs.clear()
            if self._nc:
                await self._nc.drain()
                self._nc = None

    async def _on_nats_message(self, raw_msg) -> None:
        """Handle incoming NATS message and forward to Telegram."""
        try:
            data = json.loads(raw_msg.data.decode())
        except Exception:
            data = {"text": raw_msg.data.decode(errors="replace")}

        subject = raw_msg.subject

        # Build Telegram text
        if "embed" in data:
            text = _embed_to_text(data["embed"])
        elif "text" in data:
            text = str(data["text"])
        elif "content" in data:
            text = str(data["content"])
        elif "message" in data:
            text = str(data["message"])
        else:
            text = json.dumps(data, ensure_ascii=False)[:500]

        if not text.strip():
            return

        # Add subject tag for broadcast messages
        if subject == "mesh.broadcast":
            text = f"<i>[mesh broadcast]</i>\n{text}"

        await self._rate_limited_send(text)

    async def _rate_limited_send(self, text: str) -> None:
        now = time.monotonic()
        wait = self._rate_limit - (now - self._last_send)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_send = time.monotonic()
        await _send_message(text)

    def stop(self) -> None:
        self._running = False


# ── Convenience: post a digest embed to Telegram ─────────────────────────────

async def post_digest_to_telegram(embed: dict[str, Any]) -> bool:
    """Send a digest/briefing embed directly to Telegram (no NATS needed)."""
    if not _ENABLED:
        return False
    text = _embed_to_text(embed)
    if not text.strip():
        return False
    return await _send_message(text)
