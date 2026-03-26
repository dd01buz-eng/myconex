"""
MYCONEX Twitter/X Monitor
--------------------------
Monitors Twitter/X accounts and search terms via Nitter (self-hosted or
public instance) — no API key required when using Nitter RSS feeds.

Falls back to the Twitter API v2 if TWITTER_BEARER_TOKEN is set.

Env vars:
  TWITTER_ACCOUNTS      — comma-separated @handles to monitor (no @ needed)
  TWITTER_SEARCHES      — comma-separated search terms
  TWITTER_NITTER_URL    — Nitter instance base URL (default: https://nitter.net)
  TWITTER_BEARER_TOKEN  — Twitter API v2 bearer token (optional, enables API mode)
  TWITTER_INGEST_INTERVAL — polling interval minutes (default: 60)
  TWITTER_MAX_PER_SOURCE  — max tweets per account/search per poll (default: 10)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE         = Path.home() / ".myconex"
_TW_FILE      = _BASE / "twitter_insights.json"
_TW_STAMP     = _BASE / "twitter_last_poll.json"

_ACCOUNTS     = [a.strip().lstrip("@") for a in os.getenv("TWITTER_ACCOUNTS", "").split(",") if a.strip()]
_SEARCHES     = [s.strip() for s in os.getenv("TWITTER_SEARCHES", "").split(",") if s.strip()]
_NITTER_URL   = os.getenv("TWITTER_NITTER_URL", "https://nitter.net").rstrip("/")
_BEARER       = os.getenv("TWITTER_BEARER_TOKEN", "")
_INTERVAL     = int(os.getenv("TWITTER_INGEST_INTERVAL", "60"))
_MAX_PER      = int(os.getenv("TWITTER_MAX_PER_SOURCE", "10"))


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save(path: Path, data: Any) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Nitter RSS (no auth) ──────────────────────────────────────────────────────

def _fetch_nitter_rss(url: str) -> list[dict]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MYCONEX/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        xml_text = r.read().decode("utf-8", errors="replace")

    root = ET.fromstring(xml_text)
    ns   = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    tweets = []
    for item in items[:_MAX_PER]:
        title   = item.findtext("title") or item.findtext("atom:title", namespaces=ns) or ""
        link    = item.findtext("link") or item.findtext("atom:link", namespaces=ns) or ""
        pub     = item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns) or ""
        desc    = item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or ""
        # Strip HTML tags from description
        clean   = re.sub(r"<[^>]+>", " ", desc).strip()
        text    = clean or title
        tweet_id = link.rstrip("/").split("/")[-1] if link else ""
        tweets.append({
            "id":    tweet_id,
            "text":  text[:500],
            "url":   link,
            "ts":    pub,
            "title": title[:200],
        })
    return tweets


async def _nitter_account(handle: str) -> list[dict]:
    url = f"{_NITTER_URL}/{handle}/rss"
    def _fetch():
        tweets = _fetch_nitter_rss(url)
        for t in tweets:
            t["source_name"] = f"@{handle}"
            t["source"]      = "twitter"
        return tweets
    try:
        return await asyncio.get_event_loop().run_in_executor(None, _fetch)
    except Exception as exc:
        logger.debug("[twitter] nitter fetch for @%s failed: %s", handle, exc)
        return []


async def _nitter_search(query: str) -> list[dict]:
    encoded = urllib.parse.quote(query)
    url = f"{_NITTER_URL}/search/rss?q={encoded}"
    def _fetch():
        tweets = _fetch_nitter_rss(url)
        for t in tweets:
            t["source_name"] = f"search:{query}"
            t["source"]      = "twitter"
        return tweets
    try:
        return await asyncio.get_event_loop().run_in_executor(None, _fetch)
    except Exception as exc:
        logger.debug("[twitter] nitter search for %r failed: %s", query, exc)
        return []


# ── Twitter API v2 (needs bearer token) ──────────────────────────────────────

async def _api_user_timeline(username: str) -> list[dict]:
    def _fetch():
        # First get user ID
        req = urllib.request.Request(
            f"https://api.twitter.com/2/users/by/username/{username}?user.fields=id",
            headers={"Authorization": f"Bearer {_BEARER}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            uid = json.loads(r.read())["data"]["id"]
        # Then get timeline
        params = urllib.parse.urlencode({
            "max_results": str(_MAX_PER),
            "tweet.fields": "created_at,text",
            "exclude": "retweets,replies",
        })
        req2 = urllib.request.Request(
            f"https://api.twitter.com/2/users/{uid}/tweets?{params}",
            headers={"Authorization": f"Bearer {_BEARER}"},
        )
        with urllib.request.urlopen(req2, timeout=10) as r:
            return json.loads(r.read()).get("data", [])
    try:
        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch)
        return [{
            "id":          t.get("id", ""),
            "text":        t.get("text", "")[:500],
            "url":         f"https://twitter.com/{username}/status/{t.get('id','')}",
            "ts":          t.get("created_at", ""),
            "source_name": f"@{username}",
            "source":      "twitter",
            "title":       t.get("text", "")[:80],
        } for t in raw]
    except Exception as exc:
        logger.debug("[twitter] API timeline for @%s failed: %s", username, exc)
        return []


# ── Main ingester ─────────────────────────────────────────────────────────────

class TwitterMonitor:
    def __init__(self) -> None:
        self._stamps: dict[str, str] = _load(_TW_STAMP, {})

    async def run_forever(self) -> None:
        if not _ACCOUNTS and not _SEARCHES:
            logger.info("[twitter] no TWITTER_ACCOUNTS or TWITTER_SEARCHES set — skipping")
            return
        logger.info("[twitter] monitor started — %d account(s), %d search(es), interval=%dm",
                    len(_ACCOUNTS), len(_SEARCHES), _INTERVAL)
        while True:
            await self._poll()
            await asyncio.sleep(_INTERVAL * 60)

    async def _poll(self) -> None:
        all_tweets: list[dict] = []

        # Accounts
        for handle in _ACCOUNTS:
            if _BEARER:
                tweets = await _api_user_timeline(handle)
            else:
                tweets = await _nitter_account(handle)
            all_tweets.extend(tweets)

        # Searches
        for query in _SEARCHES:
            tweets = await _nitter_search(query)
            all_tweets.extend(tweets)

        if not all_tweets:
            return

        # Deduplicate against stored
        existing  = _load(_TW_FILE, [])
        known_ids = {e.get("id") for e in existing}
        new       = [t for t in all_tweets if t.get("id") and t["id"] not in known_ids]

        if not new:
            return

        logger.info("[twitter] %d new tweet(s)", len(new))
        existing.extend(new)
        _save(_TW_FILE, existing[-2000:])
        _save(_TW_STAMP, self._stamps)

        # Embed
        try:
            from integrations.knowledge_store import embed_and_store_batch
            embed_queue = [
                {"text": t["text"], "source": "twitter",
                 "title": t.get("title", t["text"][:60]),
                 "topics": [], "ts": t.get("ts", "")}
                for t in new
            ]
            await embed_and_store_batch(embed_queue)
        except Exception as exc:
            logger.warning("[twitter] embedding failed: %s", exc)

        # Signal detector hook
        try:
            from integrations.knowledge_store import search
            from core.notifications import push as _push
            for t in new[:3]:
                if len(t["text"]) > 80:
                    _push(f"[twitter] {t.get('source_name','')} — {t['text'][:120]}")
        except Exception:
            pass
