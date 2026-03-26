"""
MYCONEX Email Ingester
-----------------------
Background service that polls Gmail, runs Fabric patterns on each new email,
and stores the extracted wisdom in ~/.myconex/ for Buzlock to use.

Two-pass extraction per email:
  Pass 1 — Fabric patterns (configurable via EMAIL_FABRIC_PATTERNS):
              extract_wisdom   → IDEAS, INSIGHTS, QUOTES, RECOMMENDATIONS, FACTS
              extract_insights → deeper insight extraction
              summarize        → one-paragraph summary
  Pass 2 — Structured JSON profile extraction (topics, likes, dislikes, etc.)
             via local Ollama as fallback when Fabric is unavailable.

All output is stored in:
  ~/.myconex/email_insights.json    — per-email record with all extracted fields
  ~/.myconex/interest_profile.json  — accumulated interest profile (topic counts etc.)
  ~/.myconex/wisdom_store.json      — chronological Fabric wisdom extractions
  ~/.myconex/memory.json            — synced summary for the remember tool
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR          = Path.home() / ".myconex"
_SEEN_FILE         = _BASE_DIR / "email_seen_uids.json"
_INSIGHTS_FILE     = _BASE_DIR / "email_insights.json"
_PROFILE_FILE      = _BASE_DIR / "interest_profile.json"
_MEMORY_FILE       = _BASE_DIR / "memory.json"
_WISDOM_FILE       = _BASE_DIR / "wisdom_store.json"

# ── Fabric config ──────────────────────────────────────────────────────────────
# Comma-separated list of patterns to run on each email (in order).
_DEFAULT_FABRIC_PATTERNS = "extract_wisdom,summarize"
_EMAIL_FABRIC_PATTERNS = [
    p.strip()
    for p in os.getenv("EMAIL_FABRIC_PATTERNS", _DEFAULT_FABRIC_PATTERNS).split(",")
    if p.strip()
]
_FABRIC_ENABLED = os.getenv("EMAIL_FABRIC_ENABLED", "true").lower() != "false"

# ── LLM endpoint (Ollama fallback — always available) ─────────────────────────
_OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")
_EXTRACT_MODEL     = os.getenv("EMAIL_EXTRACT_MODEL", "llama3.1:8b")

_EXTRACT_PROMPT = """\
You are an assistant that analyses emails to build a personal interest profile.

Read the email below and return ONLY a valid JSON object with these fields:
  "summary"       : one sentence describing what this email is about
  "topics"        : list of topics/themes (e.g. ["machine learning", "photography"])
  "project_ideas" : list of project ideas inspired by this email (may be empty)
  "likes"         : things the sender or content suggests the recipient enjoys
  "dislikes"      : things the content suggests the recipient dislikes or wants to avoid
  "people"        : notable people or organisations mentioned
  "keywords"      : 3-8 important keywords

Return only the JSON object, no extra text.

EMAIL:
Subject: {subject}
From: {sender}
Date: {date}

{body}
"""


# ── Fabric section parser ──────────────────────────────────────────────────────

def _parse_fabric_sections(text: str) -> dict[str, list[str]]:
    """
    Parse Fabric markdown output into named sections.
    Handles formats like:
      ## INSIGHTS\n- item\n- item
      # IDEAS\n* item
    Returns dict of section_name → list of bullet items.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        # Section header
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip().upper()
            sections.setdefault(current, [])
        elif current and stripped.startswith(("-", "*", "•")):
            item = stripped.lstrip("-*• ").strip()
            if item:
                sections[current].append(item)
        elif current and stripped and not stripped.startswith("#"):
            # Non-bullet content line — treat as item if section is active
            sections[current].append(stripped)
    return sections


def _fabric_sections_to_profile_fields(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map Fabric section names to profile field names."""
    mapping = {
        "IDEAS":           "project_ideas",
        "INSIGHTS":        "topics",
        "RECOMMENDATIONS": "project_ideas",
        "FACTS":           "keywords",
        "QUOTES":          "keywords",
        "HABITS":          "likes",
        "ONE-SENTENCE TAKEAWAY": "summary_items",
        "SUMMARY":         "summary_items",
    }
    result: dict[str, list[str]] = {}
    for section, items in sections.items():
        field = mapping.get(section)
        if field:
            result.setdefault(field, []).extend(items)
    return result


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


# ── Fabric extraction ─────────────────────────────────────────────────────────

async def _fabric_extract(email: dict[str, Any]) -> dict[str, Any]:
    """
    Run configured Fabric patterns on the email body.
    Returns a dict with pattern_name → raw output, plus parsed sections.
    """
    if not _FABRIC_ENABLED:
        return {}

    try:
        from integrations.fabric_client import apply_pattern, is_available
        if not is_available():
            logger.debug("[email_ingester] Fabric not available — skipping fabric pass")
            return {}
    except ImportError:
        return {}

    body = email.get("body", "")[:4000]
    subject = email.get("subject", "")
    sender = email.get("from", "")
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"

    result: dict[str, Any] = {"patterns_run": [], "raw": {}, "sections": {}, "fields": {}}

    for pattern in _EMAIL_FABRIC_PATTERNS:
        try:
            output = await apply_pattern(pattern, text)
            if output:
                result["patterns_run"].append(pattern)
                result["raw"][pattern] = output
                sections = _parse_fabric_sections(output)
                result["sections"][pattern] = sections
                fields = _fabric_sections_to_profile_fields(sections)
                for field, items in fields.items():
                    result["fields"].setdefault(field, []).extend(items)
                logger.debug("[email_ingester] fabric pattern '%s' → %d chars", pattern, len(output))
        except Exception as exc:
            logger.warning("[email_ingester] fabric pattern '%s' failed: %s", pattern, exc)

    return result


def _save_wisdom_entry(email: dict[str, Any], fabric_result: dict[str, Any]) -> None:
    """Append fabric extractions to the persistent wisdom store."""
    if not fabric_result.get("patterns_run"):
        return
    store = _load_json(_WISDOM_FILE, [])
    store.append({
        "uid": email.get("uid", ""),
        "subject": email.get("subject", ""),
        "from": email.get("from", ""),
        "date": email.get("date", ""),
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "patterns": fabric_result.get("patterns_run", []),
        "raw": fabric_result.get("raw", {}),
    })
    # Keep last 500 entries to bound file size
    _save_json(_WISDOM_FILE, store[-500:])


# ── LLM extraction ────────────────────────────────────────────────────────────

async def _extract_insights(email: dict[str, Any]) -> dict[str, Any] | None:
    """Call local Ollama to extract structured insights from one email."""
    body = email.get("body", "")[:3000]  # cap to keep prompt small
    prompt = _EXTRACT_PROMPT.format(
        subject=email.get("subject", ""),
        sender=email.get("from", ""),
        date=email.get("date", ""),
        body=body,
    )
    payload = {
        "model": _EXTRACT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_OLLAMA_URL}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Ollama returned %s", resp.status)
                    return None
                data = await resp.json()
                raw = data.get("response", "")
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return None

    # Parse JSON from the LLM response (may have markdown fences)
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("No JSON found in LLM response: %s", raw[:200])
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s — raw: %s", exc, raw[:200])
        return None


# ── Profile accumulator ───────────────────────────────────────────────────────

def _update_profile(insights: dict[str, Any], fabric_fields: dict[str, list[str]] | None = None) -> None:
    """Merge new insights (and Fabric-extracted fields) into the long-running interest profile."""
    profile = _load_json(_PROFILE_FILE, {
        "topics": {},
        "project_ideas": [],
        "likes": {},
        "dislikes": {},
        "people": {},
        "keywords": {},
        "wisdom_items": [],
        "email_count": 0,
        "last_updated": "",
    })

    def _bump(counter: dict, items: list) -> None:
        for item in items:
            item = item.strip()
            if item and len(item) < 200:  # ignore pathologically long strings
                counter[item] = counter.get(item, 0) + 1

    # ── From Ollama JSON extraction ──
    _bump(profile["topics"],   insights.get("topics", []))
    _bump(profile["likes"],    insights.get("likes", []))
    _bump(profile["dislikes"], insights.get("dislikes", []))
    _bump(profile["people"],   insights.get("people", []))
    _bump(profile["keywords"], insights.get("keywords", []))

    for idea in insights.get("project_ideas", []):
        idea = idea.strip()
        if idea and idea not in profile["project_ideas"] and len(idea) < 300:
            profile["project_ideas"].append(idea)

    # ── From Fabric patterns ──
    if fabric_fields:
        _bump(profile["topics"],   fabric_fields.get("topics", []))
        _bump(profile["likes"],    fabric_fields.get("likes", []))
        _bump(profile["keywords"], fabric_fields.get("keywords", []))

        for idea in fabric_fields.get("project_ideas", []):
            idea = idea.strip()
            if idea and idea not in profile["project_ideas"] and len(idea) < 300:
                profile["project_ideas"].append(idea)

        for item in fabric_fields.get("summary_items", []):
            item = item.strip()
            if item and item not in profile["wisdom_items"] and len(item) < 500:
                profile["wisdom_items"].append(item)

    # Keep lists bounded
    profile["project_ideas"] = profile["project_ideas"][-200:]
    profile["wisdom_items"]  = profile["wisdom_items"][-500:]

    profile["email_count"] = profile.get("email_count", 0) + 1
    profile["last_updated"] = datetime.now(timezone.utc).isoformat()

    _save_json(_PROFILE_FILE, profile)

    # Mirror top interests into the shared memory store so the
    # `remember` tool can surface them in agent conversations.
    _sync_to_memory(profile)


def _sync_to_memory(profile: dict) -> None:
    """Write a compact interest summary into the shared memory store."""
    memory = _load_json(_MEMORY_FILE, {})

    def _top(counter: dict, n: int = 10) -> list[str]:
        return [k for k, _ in sorted(counter.items(), key=lambda x: -x[1])][:n]

    memory["interests_topics"]        = ", ".join(_top(profile["topics"]))
    memory["interests_likes"]         = ", ".join(_top(profile["likes"]))
    memory["interests_dislikes"]      = ", ".join(_top(profile["dislikes"]))
    memory["interests_keywords"]      = ", ".join(_top(profile["keywords"]))
    memory["interests_people"]        = ", ".join(_top(profile["people"]))
    memory["interests_project_ideas"] = " | ".join(profile["project_ideas"][-20:])
    memory["interests_wisdom"]        = " | ".join(profile.get("wisdom_items", [])[-10:])
    memory["interests_email_count"]   = str(profile["email_count"])
    memory["interests_last_updated"]  = profile["last_updated"]

    _save_json(_MEMORY_FILE, memory)


# ── Main ingester class ────────────────────────────────────────────────────────

class EmailIngester:
    """
    Polls Gmail on a schedule, extracts insights, updates the interest profile.

    Usage:
        ingester = EmailIngester(interval_minutes=30)
        asyncio.create_task(ingester.run_forever())
    """

    def __init__(
        self,
        interval_minutes: int = 30,
        batch_size: int = 20,
        folder: str = "INBOX",
    ) -> None:
        self.interval = interval_minutes * 60
        self.batch_size = batch_size
        self.folder = folder
        self._running = False

    # ── Public ────────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Poll indefinitely, sleeping interval seconds between runs."""
        self._running = True
        logger.info("[email_ingester] starting — interval=%dm", self.interval // 60)
        while self._running:
            try:
                count = await self.ingest_once()
                if count:
                    logger.info("[email_ingester] processed %d new email(s)", count)
            except Exception as exc:
                logger.warning("[email_ingester] ingest error: %s", exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    async def ingest_once(self) -> int:
        """Fetch and process new emails. Returns count of emails processed."""
        _BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Load credentials — supports multiple accounts via comma-separated lists
        raw_addresses  = os.getenv("GMAIL_ADDRESSES", "") or os.getenv("GMAIL_ADDRESS", "")
        raw_passwords  = os.getenv("GMAIL_APP_PASSWORDS", "") or os.getenv("GMAIL_APP_PASSWORD", "")
        if not raw_addresses or not raw_passwords:
            logger.debug("[email_ingester] Gmail credentials not set — skipping")
            return 0

        addresses = [a.strip() for a in raw_addresses.split(",") if a.strip()]
        passwords = [p.strip().replace(" ", "") for p in raw_passwords.split(",") if p.strip()]

        # Load seen UIDs
        seen: set[str] = set(_load_json(_SEEN_FILE, []))

        # Fetch from all accounts
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from integrations.gmail_reader import GmailReader

        all_emails: list[dict] = []
        for address, password in zip(addresses, passwords):
            try:
                reader = GmailReader(address, password)
                fetched = reader.search(
                    query="ALL",
                    folder=self.folder,
                    limit=self.batch_size,
                    unread_only=False,
                    body_limit=3000,
                )
                # Tag each email with its source account
                for e in fetched:
                    e.setdefault("account", address)
                all_emails.extend(fetched)
                logger.debug("[email_ingester] fetched %d from %s", len(fetched), address)
            except Exception as exc:
                logger.warning("[email_ingester] Gmail fetch failed (%s): %s", address, exc)

        emails = all_emails
        new_emails = [e for e in emails if e["uid"] not in seen]
        if not new_emails:
            return 0

        logger.info("[email_ingester] %d new email(s) to process", len(new_emails))
        # Load existing insights list
        insights_log: list[dict] = _load_json(_INSIGHTS_FILE, [])
        processed = 0
        embed_queue: list[dict] = []  # collected for batch embedding after the loop

        for email_dict in new_emails:
            uid = email_dict["uid"]
            logger.info(
                "[email_ingester] processing uid=%s subject=%r",
                uid, email_dict.get("subject", "")[:60],
            )

            # Pass 1 — Fabric patterns (extract_wisdom, summarize, etc.)
            fabric_result = await _fabric_extract(email_dict)
            if fabric_result.get("patterns_run"):
                logger.info(
                    "[email_ingester] fabric ran %s on uid=%s",
                    fabric_result["patterns_run"], uid,
                )
                _save_wisdom_entry(email_dict, fabric_result)

            # Knowledge graph extraction
            _kg_text = (fabric_result.get("raw", {}).get("extract_wisdom")
                        or fabric_result.get("raw", {}).get("summarize")
                        or email_dict.get("body", ""))[:1500]
            if _kg_text:
                try:
                    from core.knowledge_graph import ingest_text as _kg_ingest
                    asyncio.create_task(_kg_ingest(
                        _kg_text, source="email",
                        title=email_dict.get("subject", ""),
                    ))
                except Exception:
                    pass

            # Queue wisdom for batch embedding after the loop
            wisdom_text = (fabric_result.get("raw", {}).get("extract_wisdom")
                           or fabric_result.get("raw", {}).get("summarize", ""))
            if wisdom_text:
                embed_queue.append({
                    "text":   wisdom_text,
                    "source": "email",
                    "metadata": {
                        "subject": email_dict.get("subject", ""),
                        "from":    email_dict.get("from", ""),
                        "date":    email_dict.get("date", ""),
                        "uid":     uid,
                    },
                })

            # Pass 2 — Structured JSON extraction via Ollama (profile fields)
            insights = await _extract_insights(email_dict)

            entry: dict[str, Any] = {
                "uid": uid,
                "subject": email_dict.get("subject", ""),
                "from": email_dict.get("from", ""),
                "date": email_dict.get("date", ""),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "summary": "",
                "topics": [],
                "project_ideas": [],
                "likes": [],
                "dislikes": [],
                "people": [],
                "keywords": [],
                "fabric_patterns": fabric_result.get("patterns_run", []),
                "fabric_summary": (
                    fabric_result.get("raw", {}).get("summarize", "")[:500]
                    if "summarize" in fabric_result.get("raw", {}) else ""
                ),
            }

            if insights:
                entry.update({k: v for k, v in insights.items() if k not in entry or not entry[k]})

            _update_profile(
                insights or {},
                fabric_fields=fabric_result.get("fields"),
            )

            insights_log.append(entry)
            seen.add(uid)
            processed += 1

        # Batch embed all queued wisdom texts in one Ollama call
        if embed_queue:
            try:
                from integrations.knowledge_store import embed_and_store_batch
                await embed_and_store_batch(embed_queue)
                logger.info("[email_ingester] batch-embedded %d wisdom item(s)", len(embed_queue))
            except Exception as exc:
                logger.debug("[email_ingester] Qdrant batch embed skipped: %s", exc)

        # Persist
        _save_json(_INSIGHTS_FILE, insights_log)
        _save_json(_SEEN_FILE, list(seen))

        # Push Discord notification
        if processed:
            try:
                from core.notifications import notify
                profile = _load_json(_PROFILE_FILE, {})
                top_topics = list(sorted(profile.get("topics", {}).items(), key=lambda x: -x[1]))[:3]
                topic_str = ", ".join(t for t, _ in top_topics) if top_topics else "various topics"
                recent_ideas = profile.get("project_ideas", [])[-2:]
                idea_str = ""
                if recent_ideas:
                    idea_str = "\n💡 Recent project ideas: " + " | ".join(recent_ideas)
                await notify(
                    f"📧 **Email digest** — processed {processed} new email(s)\n"
                    f"🏷️ Top topics: {topic_str}{idea_str}"
                )
            except Exception:
                pass

        return processed

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_profile_summary() -> str:
        """Return a human-readable interest profile summary."""
        profile = _load_json(_PROFILE_FILE, None)
        if not profile:
            return "No interest profile yet — run an email ingest first."

        def _top(counter: dict, n: int = 8) -> str:
            items = sorted(counter.items(), key=lambda x: -x[1])[:n]
            return ", ".join(f"{k} ({v})" for k, v in items) or "none"

        lines = [
            f"Interest profile  (built from {profile.get('email_count', 0)} emails)",
            f"Last updated: {profile.get('last_updated', 'unknown')}",
            "",
            f"Topics:    {_top(profile.get('topics', {}))}",
            f"Likes:     {_top(profile.get('likes', {}))}",
            f"Dislikes:  {_top(profile.get('dislikes', {}))}",
            f"People:    {_top(profile.get('people', {}))}",
            f"Keywords:  {_top(profile.get('keywords', {}))}",
        ]

        ideas = profile.get("project_ideas", [])
        if ideas:
            lines += ["", "Project ideas from emails:"]
            for idea in ideas[-10:]:
                lines.append(f"  • {idea}")

        return "\n".join(lines)

    @staticmethod
    def get_recent_insights(n: int = 10) -> str:
        """Return the n most recent processed email summaries."""
        log = _load_json(_INSIGHTS_FILE, [])
        if not log:
            return "No emails have been processed yet."
        recent = log[-n:][::-1]
        lines = [f"Last {len(recent)} processed emails:\n"]
        for e in recent:
            lines.append(f"  [{e.get('date', '')}] {e.get('subject', '(no subject)')}")
            lines.append(f"    From: {e.get('from', '')}")
            if e.get("fabric_summary"):
                lines.append(f"    Summary: {e['fabric_summary']}")
            elif e.get("summary"):
                lines.append(f"    Summary: {e['summary']}")
            if e.get("topics"):
                lines.append(f"    Topics: {', '.join(e['topics'])}")
            if e.get("fabric_patterns"):
                lines.append(f"    Fabric: {', '.join(e['fabric_patterns'])}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def get_wisdom(n: int = 5, pattern: str = "extract_wisdom") -> str:
        """Return raw Fabric wisdom extractions from the n most recent emails."""
        store = _load_json(_WISDOM_FILE, [])
        if not store:
            return "No wisdom extracted yet — emails haven't been processed with Fabric yet."
        matches = [e for e in store if not pattern or pattern in e.get("patterns", [])]
        if not matches:
            return f"No entries found for pattern '{pattern}'."
        recent = matches[-n:][::-1]
        lines = [f"Fabric wisdom from {len(recent)} email(s):\n"]
        for entry in recent:
            lines.append(f"{'─' * 60}")
            lines.append(f"[{entry.get('date', '')}] {entry.get('subject', '')}")
            lines.append(f"From: {entry.get('from', '')}\n")
            raw = entry.get("raw", {})
            text = raw.get(pattern) or next(iter(raw.values()), "")
            if text:
                lines.append(text)
            lines.append("")
        return "\n".join(lines)
