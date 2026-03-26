"""
Buzlock Discord Bot - Standalone Launcher
==========================================
Launches the MYCONEX Discord Gateway as Buzlock, the user-facing persona.

Usage:
    cd ~/myconex && source venv/bin/activate && python buzlock_bot.py

The script:
  1. Loads DISCORD_BOT_TOKEN from .env (dotenv or manual parse).
  2. Configures the DiscordGateway with application_id.
  3. Starts the gateway (connects to Discord, registers events/slash-commands).
  4. Runs until interrupted (Ctrl-C).
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# -- Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("buzlock")

# -- Constants
BOT_PERSONA = "Buzlock"
APPLICATION_ID = "1485854217141747712"


async def _supervise(name: str, coro_fn, restart_delay: int = 60) -> None:
    """
    Run coro_fn() and automatically restart it if it crashes unexpectedly.
    Propagates CancelledError so the task stops cleanly on shutdown.
    """
    while True:
        try:
            await coro_fn()
            return  # clean exit (e.g. no feeds configured) — don't restart
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[supervisor] %s crashed: %s — restarting in %ds",
                name, exc, restart_delay,
            )
            await asyncio.sleep(restart_delay)


async def main() -> None:
    """Entry-point: load config, create gateway, run until interrupted."""

    # 1. Load .env into os.environ via the unified config loader
    from config import load_config
    load_config(env_file=str(PROJECT_ROOT / ".env"))
    logger.info("Loaded .env from %s", PROJECT_ROOT / ".env")

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.error("DISCORD_BOT_TOKEN is not set - check %s", env_path)
        sys.exit(1)
    logger.info("Token loaded (prefix: %s...)", token[:20])

    # 2. Build a minimal config dict matching what DiscordGateway expects
    config: dict = {
        "discord": {
            "application_id": APPLICATION_ID,
        },
    }

    # 3. Import and instantiate the gateway
    try:
        from core.gateway.discord_gateway import DiscordGateway
    except ImportError as exc:
        logger.error("Cannot import DiscordGateway: %s", exc)
        logger.error("Make sure you run this from ~/myconex with the venv active.")
        sys.exit(1)

    gateway = DiscordGateway(config, router=None)
    logger.info(
        "DiscordGateway created - persona=%s, app_id=%s",
        BOT_PERSONA,
        APPLICATION_ID,
    )

    # 5. Graceful shutdown on SIGINT / SIGTERM
    stop_event = asyncio.Event()

    def _handle_signal(sig):
        logger.info("Received signal %s - shutting down %s...", sig, BOT_PERSONA)
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))

    # 6. Start the gateway (connects to Discord)
    logger.info("Starting %s (MYCONEX Discord Gateway)...", BOT_PERSONA)
    try:
        await gateway.start()
    except Exception as exc:
        logger.error("Failed to start gateway: %s", exc)
        sys.exit(1)

    logger.info("%s is online! Press Ctrl-C to stop.", BOT_PERSONA)

    # Warn if home channel not configured — ingester notifications won't reach Discord
    _home_ch = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
    if not _home_ch or not _home_ch.lstrip("-").isdigit():
        logger.warning(
            "DISCORD_HOME_CHANNEL is not set in .env — "
            "ingester digest notifications will be queued but never delivered. "
            "Set it to a numeric Discord channel ID."
        )

    # 6b. Start email ingester background task (if Gmail credentials are set)
    ingester_task = None
    if (os.getenv("GMAIL_ADDRESSES") or os.getenv("GMAIL_ADDRESS")) and \
       (os.getenv("GMAIL_APP_PASSWORDS") or os.getenv("GMAIL_APP_PASSWORD")):
        try:
            from integrations.email_ingester import EmailIngester
            ingester = EmailIngester(
                interval_minutes=int(os.getenv("EMAIL_INGEST_INTERVAL", "30")),
                batch_size=int(os.getenv("EMAIL_INGEST_BATCH", "20")),
            )
            ingester_task = asyncio.create_task(_supervise("email_ingester", ingester.run_forever))
            logger.info(
                "Email ingester started — polling every %s min",
                os.getenv("EMAIL_INGEST_INTERVAL", "30"),
            )
        except Exception as exc:
            logger.warning("Could not start email ingester: %s", exc)
    else:
        logger.info("Email ingester skipped — GMAIL_ADDRESSES/GMAIL_APP_PASSWORDS not set")

    # 6c. Start YouTube ingester background task (if watch history path is set)
    yt_ingester_task = None
    if os.getenv("YOUTUBE_WATCH_HISTORY_PATH"):
        try:
            from integrations.youtube_ingester import YouTubeIngester
            yt_ingester = YouTubeIngester(
                interval_minutes=int(os.getenv("YOUTUBE_INGEST_INTERVAL", "60")),
                batch_size=int(os.getenv("YOUTUBE_INGEST_BATCH", "10")),
            )
            yt_ingester_task = asyncio.create_task(_supervise("youtube_ingester", yt_ingester.run_forever))
            logger.info(
                "YouTube ingester started — polling every %s min",
                os.getenv("YOUTUBE_INGEST_INTERVAL", "60"),
            )
        except Exception as exc:
            logger.warning("Could not start YouTube ingester: %s", exc)
    else:
        logger.info("YouTube ingester skipped — YOUTUBE_WATCH_HISTORY_PATH not set")

    # 6d. Start RSS monitor background task
    rss_task = None
    try:
        from integrations.rss_monitor import RSSMonitor
        rss_monitor = RSSMonitor()
        if rss_monitor.list_feeds():
            rss_task = asyncio.create_task(_supervise("rss_monitor", rss_monitor.run_forever))
            logger.info(
                "RSS monitor started — %d feed(s), polling every %s min",
                len(rss_monitor.list_feeds()),
                os.getenv("RSS_INGEST_INTERVAL", "60"),
            )
        else:
            logger.info("RSS monitor skipped — no feeds configured (set RSS_FEEDS in .env)")
    except Exception as exc:
        logger.warning("Could not start RSS monitor: %s", exc)

    # 6e. Start podcast ingester background task
    podcast_task = None
    try:
        from integrations.podcast_ingester import PodcastIngester
        podcast_ingester = PodcastIngester()
        if podcast_ingester.list_feeds():
            podcast_task = asyncio.create_task(_supervise("podcast_ingester", podcast_ingester.run_forever))
            logger.info(
                "Podcast ingester started — %d feed(s), polling every %s min",
                len(podcast_ingester.list_feeds()),
                os.getenv("PODCAST_INGEST_INTERVAL", "120"),
            )
        else:
            logger.info("Podcast ingester skipped — no feeds configured (set PODCAST_FEEDS in .env)")
    except Exception as exc:
        logger.warning("Could not start podcast ingester: %s", exc)

    # 6f. Start cross-source signal detector
    signal_task = None
    try:
        from integrations.signal_detector import SignalDetector
        signal_detector = SignalDetector()
        signal_task = asyncio.create_task(_supervise("signal_detector", signal_detector.run_forever))
        logger.info("Signal detector started — checking every %sh", os.getenv("SIGNAL_INTERVAL_HOURS", "6"))
    except Exception as exc:
        logger.warning("Could not start signal detector: %s", exc)

    # 6l. Start plugin loader with hot-reload watcher
    plugin_loader = None
    try:
        from core.plugin_loader import PluginLoader
        _plugins_dir = PROJECT_ROOT / "plugins"
        _plugins_dir.mkdir(exist_ok=True)
        plugin_loader = PluginLoader(plugin_dir=str(_plugins_dir), hot_reload=True)
        loaded = await plugin_loader.load_all()
        logger.info("Plugin loader started — %d plugin(s) loaded, hot-reload active", len(loaded))
    except Exception as exc:
        logger.warning("Could not start plugin loader: %s", exc)

    # 6k. Start memory distiller (weekly higher-level abstraction)
    distiller_task = None
    try:
        from core.memory.distiller import MemoryDistiller
        distiller = MemoryDistiller()
        distiller_task = asyncio.create_task(_supervise("distiller", distiller.run_forever))
        logger.info("Memory distiller started — interval=%sd", os.getenv("DISTILLER_INTERVAL_DAYS", "7"))
    except Exception as exc:
        logger.warning("Could not start memory distiller: %s", exc)

    # 6j. Start exo pool consensus loop (distributed inference, consensus-gated)
    exo_task = None
    try:
        from core.exo_pool import get_pool as _get_exo_pool
        exo_pool = _get_exo_pool()
        exo_task = asyncio.create_task(_supervise("exo_pool", exo_pool.run_consensus_loop))
        logger.info(
            "Exo pool started — endpoint=%s threshold=%.2f quorum=%d",
            os.getenv("EXO_BASE_URL", "http://localhost:52415/v1"),
            float(os.getenv("EXO_COMPLEXITY_THRESHOLD", "0.85")),
            int(os.getenv("EXO_MIN_CONSENSUS_NODES", "2")),
        )
    except Exception as exc:
        logger.warning("Could not start exo pool: %s", exc)

    # 6i. Start memory consolidator (TTL pruning + near-dup removal)
    consolidator_task = None
    try:
        from core.memory.consolidator import MemoryConsolidator
        consolidator = MemoryConsolidator()
        consolidator_task = asyncio.create_task(_supervise("consolidator", consolidator.run_forever))
        logger.info("Memory consolidator started — interval=%sh", os.getenv("CONSOLIDATOR_INTERVAL_HOURS", "24"))
    except Exception as exc:
        logger.warning("Could not start memory consolidator: %s", exc)

    # 6h. Start NATS remote handler (mesh task receiver)
    nats_task = None
    if os.getenv("NATS_URL"):
        try:
            from core.messaging.remote_handler import NATSRemoteHandler
            nats_handler = NATSRemoteHandler()
            nats_task = asyncio.create_task(_supervise("nats_handler", nats_handler.run_forever))
            logger.info("NATS remote handler started — url=%s", os.getenv("NATS_URL"))
        except Exception as exc:
            logger.warning("Could not start NATS remote handler: %s", exc)
    else:
        logger.info("NATS remote handler skipped — NATS_URL not set")

    # 6g. Start web dashboard
    dashboard_task = None
    _dashboard_port = int(os.getenv("DASHBOARD_PORT", "7860"))
    _dashboard_host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    if os.getenv("DASHBOARD_ENABLED", "true").lower() != "false":
        try:
            from dashboard.app import start_dashboard
            dashboard_task = asyncio.create_task(
                _supervise("dashboard", lambda: start_dashboard(_dashboard_host, _dashboard_port))
            )
            logger.info("Dashboard started — http://%s:%d", _dashboard_host, _dashboard_port)
        except Exception as exc:
            logger.warning("Could not start dashboard: %s", exc)

    # 6m. Start Telegram bridge (NATS-to-Telegram notification forwarding)
    telegram_task = None
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        try:
            from integrations.telegram_bridge import TelegramBridge
            telegram_bridge = TelegramBridge()
            telegram_task = asyncio.create_task(_supervise("telegram_bridge", telegram_bridge.run_forever))
            logger.info("Telegram bridge started — chat_id=%s", os.getenv("TELEGRAM_CHAT_ID"))
        except Exception as exc:
            logger.warning("Could not start Telegram bridge: %s", exc)
    else:
        logger.info("Telegram bridge skipped — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable")

    # 7. Wait for shutdown signal
    await stop_event.wait()

    # 8. Clean shutdown
    if ingester_task:
        ingester_task.cancel()
        try:
            await ingester_task
        except asyncio.CancelledError:
            pass

    if yt_ingester_task:
        yt_ingester_task.cancel()
        try:
            await yt_ingester_task
        except asyncio.CancelledError:
            pass

    if rss_task:
        rss_task.cancel()
        try:
            await rss_task
        except asyncio.CancelledError:
            pass

    if podcast_task:
        podcast_task.cancel()
        try:
            await podcast_task
        except asyncio.CancelledError:
            pass

    if signal_task:
        signal_task.cancel()
        try:
            await signal_task
        except asyncio.CancelledError:
            pass

    if distiller_task:
        distiller_task.cancel()
        try:
            await distiller_task
        except asyncio.CancelledError:
            pass

    if exo_task:
        exo_task.cancel()
        try:
            await exo_task
        except asyncio.CancelledError:
            pass

    if consolidator_task:
        consolidator_task.cancel()
        try:
            await consolidator_task
        except asyncio.CancelledError:
            pass

    if nats_task:
        nats_task.cancel()
        try:
            await nats_task
        except asyncio.CancelledError:
            pass

    if dashboard_task:
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass

    if telegram_task:
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass

    if plugin_loader:
        try:
            await plugin_loader.teardown_all()
        except Exception:
            pass

    logger.info("Stopping %s gateway...", BOT_PERSONA)
    try:
        await gateway.stop()
    except Exception as exc:
        logger.warning("Error during shutdown: %s", exc)
    logger.info("%s shut down cleanly.", BOT_PERSONA)


if __name__ == "__main__":
    print()
    print("=" * 50)
    print(f"  {BOT_PERSONA} - MYCONEX Discord Bot")
    print("=" * 50)
    print()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"{BOT_PERSONA} stopped by user.")
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
