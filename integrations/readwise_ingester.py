"""
MYCONEX Readwise + Pocket Ingester
------------------------------------
Pulls highlights from Readwise and saved articles from Pocket,
embeds them into the knowledge base.

Readwise env vars:
  READWISE_TOKEN        — API token from readwise.io/access_token
  READWISE_INGEST_INTERVAL — polling interval minutes (default: 120)

Pocket env vars:
  POCKET_CONSUMER_KEY  — app consumer key
  POCKET_ACCESS_TOKEN  — user access token
  POCKET_INGEST_INTERVAL — polling interval minutes (default: 120)

Getting Pocket tokens:
  1. Create an app at getpocket.com/developer
  2. Use the OAuth flow or a tool like pocket-auth to get the access token
  3. Set POCKET_CONSUMER_KEY + POCKET_ACCESS_TOKEN in .env
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE = Path.home() / ".myconex"
_RW_FILE     = _BASE / "readwise_insights.json"
_POCKET_FILE = _BASE / "pocket_insights.json"
_RW_STAMP    = _BASE / "readwise_last_poll.txt"
_POCKET_STAMP = _BASE / "pocket_last_poll.txt"

_RW_TOKEN    = os.getenv("READWISE_TOKEN", "")
_RW_INTERVAL = int(os.getenv("READWISE_INGEST_INTERVAL", "120"))

_PK_KEY      = os.getenv("POCKET_CONSUMER_KEY", "")
_PK_TOKEN    = os.getenv("POCKET_ACCESS_TOKEN", "")
_PK_INTERVAL = int(os.getenv("POCKET_INGEST_INTERVAL", "120"))


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


# ── Readwise ──────────────────────────────────────────────────────────────────

def _rw_request(endpoint: str, params: dict | None = None) -> Any:
    url = f"https://readwise.io/api/v2/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Token {_RW_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


async def _fetch_readwise_highlights(updated_after: str | None) -> list[dict]:
    params: dict = {"page_size": "100"}
    if updated_after:
        params["updated__gt"] = updated_after

    def _fetch():
        results = []
        url_params = dict(params)
        while True:
            data = _rw_request("highlights", url_params)
            for h in data.get("results", []):
                book_title = h.get("book", {}).get("title", "") if isinstance(h.get("book"), dict) else ""
                results.append({
                    "id":          str(h.get("id", "")),
                    "text":        h.get("text", ""),
                    "note":        h.get("note", ""),
                    "book_title":  book_title,
                    "source_url":  h.get("url", ""),
                    "updated_at":  h.get("updated", ""),
                    "highlighted_at": h.get("highlighted_at", ""),
                    "source":      "readwise",
                    "topics":      [],
                })
            next_page = data.get("next")
            if not next_page:
                break
            # parse next cursor from URL
            parsed = urllib.parse.urlparse(next_page)
            url_params = dict(urllib.parse.parse_qsl(parsed.query))
        return results

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


class ReadwiseIngester:
    async def run_forever(self) -> None:
        if not _RW_TOKEN:
            logger.info("[readwise] READWISE_TOKEN not set — skipping")
            return
        logger.info("[readwise] ingester started — interval=%dm", _RW_INTERVAL)
        while True:
            await self._poll()
            await asyncio.sleep(_RW_INTERVAL * 60)

    async def _poll(self) -> None:
        stamp = _RW_STAMP.read_text().strip() if _RW_STAMP.exists() else None
        try:
            highlights = await _fetch_readwise_highlights(stamp)
        except Exception as exc:
            logger.warning("[readwise] fetch failed: %s", exc)
            return

        if not highlights:
            return

        existing = _load(_RW_FILE, [])
        known_ids = {e.get("id") for e in existing}
        new = [h for h in highlights if h["id"] not in known_ids]

        if not new:
            return

        logger.info("[readwise] %d new highlight(s)", len(new))
        existing.extend(new)
        _save(_RW_FILE, existing[-2000:])

        # Update stamp
        latest = max(h["updated_at"] for h in new if h["updated_at"])
        if latest:
            _BASE.mkdir(parents=True, exist_ok=True)
            _RW_STAMP.write_text(latest)

        # Embed
        try:
            from integrations.knowledge_store import embed_and_store_batch
            embed_queue = [
                {
                    "text":   f"{h['book_title']}\n{h['text']}" + (f"\nNote: {h['note']}" if h["note"] else ""),
                    "source": "readwise",
                    "title":  h["book_title"] or "Highlight",
                    "topics": [],
                    "ts":     h["updated_at"],
                }
                for h in new
            ]
            await embed_and_store_batch(embed_queue)
        except Exception as exc:
            logger.warning("[readwise] embedding failed: %s", exc)


# ── Pocket ────────────────────────────────────────────────────────────────────

async def _fetch_pocket_articles(since: int | None) -> list[dict]:
    params: dict = {
        "consumer_key": _PK_KEY,
        "access_token": _PK_TOKEN,
        "count":        "50",
        "sort":         "newest",
        "detailType":   "complete",
        "state":        "all",
    }
    if since:
        params["since"] = str(since)

    def _fetch():
        payload = json.dumps(params).encode()
        req = urllib.request.Request(
            "https://getpocket.com/v3/get",
            data=payload,
            headers={"Content-Type": "application/json", "X-Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

    articles = []
    for item_id, item in (data.get("list") or {}).items():
        title   = item.get("resolved_title") or item.get("given_title", "")
        url     = item.get("resolved_url") or item.get("given_url", "")
        excerpt = item.get("excerpt", "")[:500]
        tags    = list(item.get("tags", {}).keys())
        added   = item.get("time_added", "0")
        articles.append({
            "id":       item_id,
            "title":    title,
            "url":      url,
            "excerpt":  excerpt,
            "tags":     tags,
            "added_at": added,
            "text":     f"{title}\n{url}\n{excerpt}",
            "source":   "pocket",
            "topics":   tags,
        })
    return articles


class PocketIngester:
    async def run_forever(self) -> None:
        if not (_PK_KEY and _PK_TOKEN):
            logger.info("[pocket] POCKET_CONSUMER_KEY / POCKET_ACCESS_TOKEN not set — skipping")
            return
        logger.info("[pocket] ingester started — interval=%dm", _PK_INTERVAL)
        while True:
            await self._poll()
            await asyncio.sleep(_PK_INTERVAL * 60)

    async def _poll(self) -> None:
        stamp_raw = _POCKET_STAMP.read_text().strip() if _POCKET_STAMP.exists() else None
        since = int(stamp_raw) if stamp_raw and stamp_raw.isdigit() else None
        try:
            articles = await _fetch_pocket_articles(since)
        except Exception as exc:
            logger.warning("[pocket] fetch failed: %s", exc)
            return

        if not articles:
            return

        existing = _load(_POCKET_FILE, [])
        known_ids = {e.get("id") for e in existing}
        new = [a for a in articles if a["id"] not in known_ids]

        if not new:
            return

        logger.info("[pocket] %d new article(s)", len(new))
        existing.extend(new)
        _save(_POCKET_FILE, existing[-2000:])

        import time
        _BASE.mkdir(parents=True, exist_ok=True)
        _POCKET_STAMP.write_text(str(int(time.time())))

        try:
            from integrations.knowledge_store import embed_and_store_batch
            embed_queue = [
                {"text": a["text"], "source": "pocket", "title": a["title"],
                 "topics": a["topics"], "ts": a["added_at"]}
                for a in new
            ]
            await embed_and_store_batch(embed_queue)
        except Exception as exc:
            logger.warning("[pocket] embedding failed: %s", exc)


# ── Combined runner ───────────────────────────────────────────────────────────

class ReadwisePocketIngester:
    """Runs both ingesters together as a single supervised task."""
    async def run_forever(self) -> None:
        tasks = []
        if _RW_TOKEN:
            tasks.append(asyncio.create_task(ReadwiseIngester().run_forever()))
        if _PK_KEY and _PK_TOKEN:
            tasks.append(asyncio.create_task(PocketIngester().run_forever()))
        if not tasks:
            logger.info("[readwise/pocket] no credentials set — skipping both")
            return
        await asyncio.gather(*tasks)
