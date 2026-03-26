"""
MYCONEX RSS/Atom Feed Monitor
------------------------------
Polls a configurable list of RSS and Atom feeds, runs Fabric patterns on new
articles, stores extracted wisdom in the shared knowledge base, and pushes
digest notifications to the Discord notification bus.

.env keys:
  RSS_FEEDS               — semicolon-separated list of feed URLs
  RSS_INGEST_INTERVAL     — poll interval in minutes (default 60)
  RSS_INGEST_BATCH        — max new articles to process per poll pass (default 20)
  RSS_FABRIC_PATTERNS     — comma-separated Fabric patterns (default: extract_wisdom,summarize)
  RSS_FABRIC_ENABLED      — set false to skip Fabric (default: true)
  RSS_MIN_CONTENT_LEN     — min chars for an article to be processed (default: 200)

Outputs:
  ~/.myconex/rss_seen_ids.json      — hashes of already-processed articles
  ~/.myconex/rss_insights.json      — per-article insight log
  ~/.myconex/wisdom_store.json      — shared wisdom store (also used by email/youtube)
  ~/.myconex/interest_profile.json  — shared interest profile

Usage:
    from integrations.rss_monitor import RSSMonitor
    monitor = RSSMonitor()
    asyncio.create_task(monitor.run_forever())

To add feeds at runtime (Discord `rss_add` tool):
    monitor.add_feed("https://example.com/feed.xml")

To list feeds:
    monitor.list_feeds()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR      = Path.home() / ".myconex"
_SEEN_FILE     = _BASE_DIR / "rss_seen_ids.json"
_INSIGHTS_FILE = _BASE_DIR / "rss_insights.json"
_WISDOM_FILE   = _BASE_DIR / "wisdom_store.json"
_PROFILE_FILE  = _BASE_DIR / "interest_profile.json"
_FEEDS_FILE    = _BASE_DIR / "rss_feeds.json"

# ── Config ─────────────────────────────────────────────────────────────────────
_INGEST_INTERVAL  = int(os.getenv("RSS_INGEST_INTERVAL", "60"))
_BATCH_SIZE       = int(os.getenv("RSS_INGEST_BATCH", "20"))
_FABRIC_PATTERNS  = [
    p.strip()
    for p in os.getenv("RSS_FABRIC_PATTERNS", "extract_wisdom,summarize").split(",")
    if p.strip()
]
_FABRIC_ENABLED   = os.getenv("RSS_FABRIC_ENABLED", "true").lower() != "false"
_MIN_CONTENT_LEN  = int(os.getenv("RSS_MIN_CONTENT_LEN", "200"))

# Feeds from env (semicolon-separated)
_ENV_FEEDS = [
    f.strip()
    for f in os.getenv("RSS_FEEDS", "").split(";")
    if f.strip()
]


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save_json(path: Path, data: Any) -> None:
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _article_id(url: str, title: str) -> str:
    """Stable hash for an article — URL is primary, title is fallback."""
    key = url.strip() if url.strip() else title.strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── feedparser wrapper ─────────────────────────────────────────────────────────

async def _fetch_feed(url: str) -> list[dict[str, Any]]:
    """
    Fetch and parse a feed URL.  Returns a list of article dicts with:
    id, url, title, content, published, feed_url.
    """
    try:
        import feedparser  # type: ignore[import]
    except ImportError:
        logger.warning("[rss] feedparser not installed — run: pip install feedparser")
        return []

    try:
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, url)
    except Exception as exc:
        logger.warning("[rss] fetch failed for %s: %s", url, exc)
        return []

    articles: list[dict[str, Any]] = []
    for entry in feed.get("entries", []):
        link   = entry.get("link", "")
        title  = entry.get("title", "")
        # Prefer full content, fall back to summary
        content = ""
        if entry.get("content"):
            content = entry["content"][0].get("value", "")
        if not content:
            content = entry.get("summary", "")
        # Strip basic HTML tags for text extraction
        import re
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

        published = ""
        if entry.get("published"):
            published = entry["published"]
        elif entry.get("updated"):
            published = entry["updated"]

        art_id = _article_id(link, title)
        articles.append({
            "id":        art_id,
            "url":       link,
            "title":     title,
            "content":   content,
            "published": published,
            "feed_url":  url,
            "feed_title": feed.get("feed", {}).get("title", urlparse(url).netloc),
        })

    return articles


# ── Fabric processing ──────────────────────────────────────────────────────────

async def _fabric_process(text: str, title: str) -> dict[str, Any]:
    """Run configured Fabric patterns on article text. Returns result dict."""
    if not _FABRIC_ENABLED or not text.strip():
        return {}

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from integrations.fabric_client import apply_pattern, is_available
        if not is_available():
            return {}
    except ImportError as exc:
        logger.warning("[rss] fabric_client import error: %s", exc)
        return {}

    result: dict[str, Any] = {"patterns_run": [], "raw": {}}
    for pattern in _FABRIC_PATTERNS:
        try:
            output = await apply_pattern(pattern, text)
            if output:
                result["patterns_run"].append(pattern)
                result["raw"][pattern] = output
        except Exception as exc:
            logger.warning("[rss] pattern '%s' failed for '%s': %s", pattern, title[:50], exc)

    return result


# ── Profile / wisdom update ────────────────────────────────────────────────────

def _parse_fabric_sections(text: str) -> dict[str, list[str]]:
    """Parse markdown-ish Fabric output into named sections."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip().upper()
            sections.setdefault(current, [])
        elif current and stripped.startswith(("-", "*", "•")):
            item = stripped.lstrip("-*• ").strip()
            if item:
                sections[current].append(item)
        elif current and stripped and not stripped.startswith("#"):
            sections[current].append(stripped)
    return sections


def _update_profile_from_article(
    fabric_result: dict[str, Any], title: str, feed_title: str
) -> None:
    """Merge article wisdom into the shared interest profile."""
    profile = _load_json(_PROFILE_FILE, {
        "topics": {}, "project_ideas": [], "likes": {}, "dislikes": {},
        "people": {}, "keywords": {}, "wisdom_items": [],
        "email_count": 0, "video_count": 0, "rss_count": 0, "last_updated": "",
    })

    section_map = {
        "IDEAS":                  "project_ideas",
        "INSIGHTS":               "topics",
        "RECOMMENDATIONS":        "project_ideas",
        "FACTS":                  "keywords",
        "QUOTES":                 "keywords",
        "ONE-SENTENCE TAKEAWAY":  "wisdom_items",
        "SUMMARY":                "wisdom_items",
    }

    def _bump(counter: dict, items: list) -> None:
        for item in items:
            item = item.strip()
            if item and len(item) < 200:
                counter[item] = counter.get(item, 0) + 1

    for pattern_output in fabric_result.get("raw", {}).values():
        sections = _parse_fabric_sections(pattern_output)
        for section, items in sections.items():
            field = section_map.get(section)
            if not field:
                continue
            if field in ("topics", "keywords", "likes"):
                _bump(profile[field], items)
            elif field == "project_ideas":
                for idea in items:
                    idea = idea.strip()
                    if idea and idea not in profile["project_ideas"] and len(idea) < 300:
                        profile["project_ideas"].append(idea)
            elif field == "wisdom_items":
                for item in items:
                    item = item.strip()
                    if item and item not in profile["wisdom_items"] and len(item) < 500:
                        profile["wisdom_items"].append(item)

    profile["project_ideas"] = profile["project_ideas"][-200:]
    profile["wisdom_items"]  = profile["wisdom_items"][-500:]
    profile["rss_count"]     = profile.get("rss_count", 0) + 1
    profile["last_updated"]  = datetime.now(timezone.utc).isoformat()
    _save_json(_PROFILE_FILE, profile)


def _save_wisdom_entry(article: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    if not fabric_result.get("patterns_run"):
        return
    store = _load_json(_WISDOM_FILE, [])
    store.append({
        "source":     "rss",
        "article_id": article["id"],
        "url":        article["url"],
        "title":      article["title"],
        "feed_title": article.get("feed_title", ""),
        "published":  article.get("published", ""),
        "stored_at":  datetime.now(timezone.utc).isoformat(),
        "patterns":   fabric_result.get("patterns_run", []),
        "raw":        fabric_result.get("raw", {}),
    })
    _save_json(_WISDOM_FILE, store[-500:])


def _save_insight_entry(article: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    log = _load_json(_INSIGHTS_FILE, [])
    log.append({
        "article_id":      article["id"],
        "url":             article["url"],
        "title":           article["title"],
        "feed_title":      article.get("feed_title", ""),
        "feed_url":        article.get("feed_url", ""),
        "published":       article.get("published", ""),
        "processed_at":    datetime.now(timezone.utc).isoformat(),
        "fabric_patterns": fabric_result.get("patterns_run", []),
        "summary":         fabric_result.get("raw", {}).get("summarize", "")[:500],
        "content_length":  len(article.get("content", "")),
    })
    _save_json(_INSIGHTS_FILE, log[-1000:])


# ── Main monitor class ─────────────────────────────────────────────────────────

class RSSMonitor:
    """
    Polls RSS/Atom feeds, processes new articles through Fabric, and stores
    the extracted wisdom in the shared knowledge base.

    Usage:
        monitor = RSSMonitor()
        asyncio.create_task(monitor.run_forever())
    """

    def __init__(
        self,
        feeds: list[str] | None = None,
        interval_minutes: int = _INGEST_INTERVAL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        # Load persisted feeds from disk, then merge with env + constructor feeds
        persisted: list[str] = _load_json(_FEEDS_FILE, [])
        combined = list(dict.fromkeys(
            persisted + _ENV_FEEDS + (feeds or [])
        ))
        self._feeds: list[str] = combined
        self.interval  = interval_minutes * 60
        self.batch_size = batch_size
        self._running  = False

    # ── Feed management ───────────────────────────────────────────────────────

    def add_feed(self, url: str) -> bool:
        """Add a feed URL. Returns True if newly added."""
        url = url.strip()
        if not url or url in self._feeds:
            return False
        self._feeds.append(url)
        _save_json(_FEEDS_FILE, self._feeds)
        logger.info("[rss] added feed: %s", url)
        return True

    def remove_feed(self, url: str) -> bool:
        """Remove a feed URL. Returns True if it was present."""
        url = url.strip()
        if url in self._feeds:
            self._feeds.remove(url)
            _save_json(_FEEDS_FILE, self._feeds)
            logger.info("[rss] removed feed: %s", url)
            return True
        return False

    def list_feeds(self) -> list[str]:
        return list(self._feeds)

    # ── Background task ───────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        if not self._feeds:
            logger.info("[rss] no feeds configured — background polling disabled")
            return
        self._running = True
        logger.info("[rss] starting — %d feed(s), interval=%dm", len(self._feeds), self.interval // 60)
        while self._running:
            try:
                count = await self.poll_all()
                if count:
                    logger.info("[rss] processed %d new article(s)", count)
            except Exception as exc:
                logger.warning("[rss] poll error: %s", exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    # ── Poll all feeds ────────────────────────────────────────────────────────

    async def poll_all(self) -> int:
        """Fetch all feeds and process new articles. Returns total processed count."""
        if not self._feeds:
            return 0

        seen: set[str] = set(_load_json(_SEEN_FILE, []))
        all_new: list[dict[str, Any]] = []

        # Fetch all feeds concurrently
        fetch_tasks = [_fetch_feed(url) for url in self._feeds]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for feed_url, articles in zip(self._feeds, results):
            if isinstance(articles, Exception):
                logger.warning("[rss] error fetching %s: %s", feed_url, articles)
                continue
            for art in articles:
                if art["id"] not in seen:
                    all_new.append(art)

        # Sort oldest first, cap to batch size
        all_new.sort(key=lambda a: a.get("published", ""))
        batch = all_new[: self.batch_size]

        processed = 0
        embed_queue: list[dict] = []
        for article in batch:
            success, embed_item = await self._process_article(article)
            if success:
                processed += 1
                if embed_item:
                    embed_queue.append(embed_item)
            seen.add(article["id"])

        if batch:
            _save_json(_SEEN_FILE, list(seen))

        # Batch embed all queued wisdom texts in one Ollama call
        if embed_queue:
            try:
                from integrations.knowledge_store import embed_and_store_batch
                await embed_and_store_batch(embed_queue)
                logger.info("[rss] batch-embedded %d article(s)", len(embed_queue))
            except Exception as exc:
                logger.debug("[rss] Qdrant batch embed skipped: %s", exc)

        if processed:
            try:
                from core.notifications import notify
                profile = _load_json(_PROFILE_FILE, {})
                top_topics = sorted(profile.get("topics", {}).items(), key=lambda x: -x[1])[:3]
                topic_str = ", ".join(t for t, _ in top_topics) if top_topics else "various topics"
                # Collect unique feed names in this batch
                feed_names = list(dict.fromkeys(
                    a.get("feed_title", urlparse(a.get("feed_url", "")).netloc)
                    for a in batch[:processed]
                ))[:3]
                feeds_str = ", ".join(feed_names) if feed_names else "feeds"
                await notify(
                    f"📰 **RSS digest** — {processed} new article(s) from {feeds_str}\n"
                    f"🏷️ Top topics: {topic_str}"
                )
            except Exception:
                pass

        return processed

    # ── Single article ────────────────────────────────────────────────────────

    async def _process_article(self, article: dict[str, Any]) -> tuple[bool, dict | None]:
        """Process one article.

        Returns (success, embed_item) where embed_item is a dict ready for
        embed_and_store_batch(), or None if there is nothing to embed.
        """
        content = article.get("content", "")
        title   = article.get("title", "")[:80]

        if len(content) < _MIN_CONTENT_LEN:
            logger.debug("[rss] skipping short article (%d chars): %s", len(content), title)
            _save_insight_entry(article, {})
            return False, None

        logger.info("[rss] processing: %s  [%s]", title, article.get("feed_title", ""))
        fabric_result = await _fabric_process(content, title)
        _save_insight_entry(article, fabric_result)

        if fabric_result.get("patterns_run"):
            _save_wisdom_entry(article, fabric_result)
            _update_profile_from_article(fabric_result, title, article.get("feed_title", ""))

            wisdom_text = (
                fabric_result.get("raw", {}).get("extract_wisdom")
                or fabric_result.get("raw", {}).get("summarize", "")
            )

            # Knowledge graph extraction
            if wisdom_text:
                try:
                    from core.knowledge_graph import ingest_text as _kg_ingest
                    asyncio.ensure_future(_kg_ingest(
                        wisdom_text[:1500], source="rss",
                        title=article.get("title", ""),
                    ))
                except Exception:
                    pass

            embed_item = None
            if wisdom_text:
                embed_item = {
                    "text":   wisdom_text,
                    "source": "rss",
                    "metadata": {
                        "title":      article.get("title", ""),
                        "url":        article.get("url", ""),
                        "feed_title": article.get("feed_title", ""),
                        "published":  article.get("published", ""),
                    },
                }
            return True, embed_item
        return False, None

    # ── Summaries ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_recent_insights(n: int = 10) -> str:
        log = _load_json(_INSIGHTS_FILE, [])
        if not log:
            return "No RSS articles have been processed yet."
        recent = log[-n:][::-1]
        lines = [f"Last {len(recent)} processed article(s):\n"]
        for e in recent:
            lines.append(f"  [{e.get('feed_title', '')}]  {e.get('title', '')}")
            lines.append(f"    URL:       {e.get('url', '')}")
            lines.append(f"    Published: {e.get('published', '')}")
            if e.get("summary"):
                lines.append(f"    Summary:   {e['summary'][:200]}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def get_wisdom(n: int = 5) -> str:
        store = _load_json(_WISDOM_FILE, [])
        rss_entries = [e for e in store if e.get("source") == "rss"]
        if not rss_entries:
            return "No RSS wisdom extracted yet."
        recent = rss_entries[-n:][::-1]
        lines = [f"Fabric wisdom from {len(recent)} article(s):\n"]
        for entry in recent:
            lines.append(f"{'─' * 60}")
            lines.append(f"[{entry.get('feed_title', '')}]  {entry.get('title', '')}")
            lines.append(f"URL: {entry.get('url', '')}\n")
            raw = entry.get("raw", {})
            text = raw.get("extract_wisdom") or raw.get("summarize") or next(iter(raw.values()), "")
            if text:
                lines.append(text[:1000])
            lines.append("")
        return "\n".join(lines)
