"""Tests for MYCONEX service discovery (mDNS hub service auto-discovery)."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import ServiceURLs, ServiceDiscoveryError, MeshDiscovery, ServiceAdvertiser, ServiceWatcher, resolve_service_urls


class TestServiceURLs(unittest.TestCase):

    def test_defaults_are_none(self):
        urls = ServiceURLs()
        self.assertIsNone(urls.nats_url)
        self.assertIsNone(urls.redis_url)
        self.assertIsNone(urls.qdrant_url)

    def test_partial_population(self):
        urls = ServiceURLs(nats_url="nats://192.168.1.10:4222")
        self.assertEqual(urls.nats_url, "nats://192.168.1.10:4222")
        self.assertIsNone(urls.redis_url)

    def test_service_discovery_error_is_runtime_error(self):
        err = ServiceDiscoveryError("NATS not found")
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("NATS", str(err))


class TestMeshDiscoveryZeroconfProperty(unittest.TestCase):

    def test_zeroconf_raises_before_start(self):
        discovery = MeshDiscovery(
            node_name="test-node", tier="T3", roles=["relay"]
        )
        with self.assertRaises(RuntimeError) as ctx:
            _ = discovery.zeroconf
        self.assertIn("not started", str(ctx.exception))

    @patch('core.discovery.mesh_discovery.AsyncZeroconf')
    @patch('core.discovery.mesh_discovery.AsyncServiceBrowser')
    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_zeroconf_returns_instance_after_start(
        self, mock_info, mock_browser, mock_zc
    ):
        mock_zc_instance = AsyncMock()
        mock_zc.return_value = mock_zc_instance

        discovery = MeshDiscovery(
            node_name="test-node", tier="T3", roles=["relay"]
        )

        async def run():
            await discovery.start()
            zc = discovery.zeroconf
            self.assertIs(zc, mock_zc_instance)
            await discovery.stop()

        asyncio.run(run())


class TestServiceAdvertiser(unittest.TestCase):

    def _make_advertiser(self, mock_zc):
        """Build a ServiceAdvertiser with a fake config and zeroconf."""
        cfg = MagicMock()
        cfg.mesh.nats_url = "nats://localhost:4222"
        cfg.mesh.redis_url = "redis://localhost:6379"
        cfg.mesh.qdrant_url = "http://localhost:6333"
        return ServiceAdvertiser(zc=mock_zc, node_name="hub-node", cfg=cfg)

    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_registers_services_for_open_ports(self, mock_service_info):
        """Registers mDNS records only for ports that respond."""
        mock_zc = AsyncMock()
        mock_zc.async_register_service = AsyncMock()
        mock_si = AsyncMock()
        mock_service_info.return_value = mock_si

        advertiser = self._make_advertiser(mock_zc)

        async def run():
            # Patch _probe_port so NATS and Qdrant respond, Redis does not
            async def fake_probe(port, timeout=1.0):
                return port in (4222, 6333)

            advertiser._probe_port = fake_probe
            await advertiser.start()
            # Should register two services
            self.assertEqual(mock_zc.async_register_service.call_count, 2)

        asyncio.run(run())

    @patch('core.discovery.mesh_discovery.AsyncServiceInfo')
    def test_silent_on_all_probe_failures(self, mock_service_info):
        """No error raised if no services respond (normal for worker nodes)."""
        mock_zc = AsyncMock()
        mock_zc.async_register_service = AsyncMock()
        advertiser = self._make_advertiser(mock_zc)

        async def run():
            async def always_false(port, timeout=1.0):
                return False

            advertiser._probe_port = always_false
            await advertiser.start()  # must not raise
            mock_zc.async_register_service.assert_not_called()

        asyncio.run(run())

    def test_probe_port_returns_false_on_connection_refused(self):
        """_probe_port returns False when nothing is listening."""
        mock_zc = AsyncMock()
        cfg = MagicMock()
        cfg.mesh.nats_url = "nats://localhost:4222"
        cfg.mesh.redis_url = "redis://localhost:6379"
        cfg.mesh.qdrant_url = "http://localhost:6333"
        advertiser = ServiceAdvertiser(zc=mock_zc, node_name="test", cfg=cfg)

        async def run():
            # Port 1 will always be refused
            result = await advertiser._probe_port(1, timeout=0.1)
            self.assertFalse(result)

        asyncio.run(run())


class TestServiceWatcher(unittest.TestCase):

    def test_initial_urls_are_none(self):
        mock_zc = MagicMock()
        watcher = ServiceWatcher(zc=mock_zc)
        urls = watcher.get_urls()
        self.assertIsNone(urls.nats_url)
        self.assertIsNone(urls.redis_url)
        self.assertIsNone(urls.qdrant_url)

    def test_sync_callback_fires_on_url_change(self):
        """Sync callbacks receive updated ServiceURLs."""
        mock_zc = MagicMock()
        watcher = ServiceWatcher(zc=mock_zc)

        received = []
        watcher.on_change(lambda urls: received.append(urls))

        # Simulate an mDNS discovery event
        watcher._update_url("nats_url", "nats://192.168.1.10:4222")

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].nats_url, "nats://192.168.1.10:4222")

    def test_async_callback_fires_on_url_change(self):
        """Coroutine callbacks are awaited."""
        mock_zc = MagicMock()
        watcher = ServiceWatcher(zc=mock_zc)

        received = []

        async def async_cb(urls):
            received.append(urls)

        watcher.on_change(async_cb)
        watcher._update_url("redis_url", "redis://192.168.1.10:6379")

        async def run():
            await asyncio.sleep(0.05)  # let coroutine schedule
        asyncio.run(run())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].redis_url, "redis://192.168.1.10:6379")

    def test_url_set_to_none_on_service_removal(self):
        """Removing an mDNS record sets the URL to None and fires callbacks."""
        mock_zc = MagicMock()
        watcher = ServiceWatcher(zc=mock_zc)

        watcher._update_url("nats_url", "nats://192.168.1.10:4222")
        watcher._update_url("nats_url", None)

        urls = watcher.get_urls()
        self.assertIsNone(urls.nats_url)

    def test_partial_urls_are_valid(self):
        """A ServiceURLs with some None fields is a valid callback payload."""
        mock_zc = MagicMock()
        watcher = ServiceWatcher(zc=mock_zc)

        received = []
        watcher.on_change(lambda urls: received.append(urls))
        watcher._update_url("nats_url", "nats://192.168.1.10:4222")
        # redis and qdrant still None — callback still fires

        self.assertEqual(len(received), 1)
        self.assertIsNotNone(received[0].nats_url)
        self.assertIsNone(received[0].redis_url)


class TestResolveServiceUrls(unittest.TestCase):

    def _make_cfg(self, nats=None, redis=None, qdrant=None):
        cfg = MagicMock()
        cfg.mesh.nats_url   = nats   or "nats://localhost:4222"
        cfg.mesh.redis_url  = redis  or "redis://localhost:6379"
        cfg.mesh.qdrant_url = qdrant or "http://localhost:6333"
        return cfg

    def test_env_vars_take_priority_over_mdns(self):
        """If env var is set, skip mDNS for that service."""
        cfg = self._make_cfg()
        mock_zc = MagicMock()

        with patch.dict(os.environ, {
            "NATS_URL": "nats://myserver:4222",
            "REDIS_URL": "redis://myserver:6379",
            "QDRANT_URL": "http://myserver:6333",
        }):
            async def run():
                with patch(
                    'core.discovery.mesh_discovery.ServiceWatcher'
                ) as MockWatcher:
                    mock_watcher = MagicMock()
                    mock_watcher.get_urls = MagicMock(return_value=ServiceURLs())  # empty — mDNS found nothing
                    mock_watcher.start = AsyncMock()
                    MockWatcher.return_value = mock_watcher

                    urls, watcher = await resolve_service_urls(
                        zc=mock_zc, cfg=cfg, timeout=0.1
                    )
                    self.assertEqual(urls.nats_url, "nats://myserver:4222")
                    self.assertEqual(urls.redis_url, "redis://myserver:6379")
                    self.assertEqual(urls.qdrant_url, "http://myserver:6333")

            asyncio.run(run())

    def test_mdns_used_when_no_env_var(self):
        """mDNS result used when env var is not set."""
        cfg = self._make_cfg()
        mock_zc = MagicMock()

        env = {k: v for k, v in os.environ.items()
               if k not in ("NATS_URL", "REDIS_URL", "QDRANT_URL")}

        async def run():
            with patch.dict(os.environ, env, clear=True):
                with patch(
                    'core.discovery.mesh_discovery.ServiceWatcher'
                ) as MockWatcher:
                    mock_watcher = MagicMock()
                    mock_watcher.get_urls = MagicMock(return_value=ServiceURLs(
                        nats_url="nats://hub:4222",
                        redis_url="redis://hub:6379",
                        qdrant_url="http://hub:6333",
                    ))
                    mock_watcher.start = AsyncMock()
                    MockWatcher.return_value = mock_watcher

                    urls, watcher = await resolve_service_urls(
                        zc=mock_zc, cfg=cfg, timeout=0.1
                    )
                    self.assertEqual(urls.nats_url, "nats://hub:4222")

        asyncio.run(run())

    def test_raises_when_no_env_var_and_no_mdns(self):
        """ServiceDiscoveryError when env var absent and mDNS found nothing."""
        cfg = self._make_cfg()
        mock_zc = MagicMock()

        env = {k: v for k, v in os.environ.items()
               if k not in ("NATS_URL", "REDIS_URL", "QDRANT_URL")}

        async def run():
            with patch.dict(os.environ, env, clear=True):
                with patch(
                    'core.discovery.mesh_discovery.ServiceWatcher'
                ) as MockWatcher:
                    mock_watcher = MagicMock()
                    mock_watcher.get_urls = MagicMock(return_value=ServiceURLs())  # nothing found
                    mock_watcher.start = AsyncMock()
                    MockWatcher.return_value = mock_watcher

                    with self.assertRaises(ServiceDiscoveryError) as ctx:
                        await resolve_service_urls(
                            zc=mock_zc, cfg=cfg, timeout=0.1
                        )
                    self.assertIn("NATS", str(ctx.exception))
                    self.assertIn("Fix:", str(ctx.exception))

        asyncio.run(run())

    def test_returns_live_watcher(self):
        """resolve_service_urls returns a live ServiceWatcher for reconnect use."""
        cfg = self._make_cfg()
        mock_zc = MagicMock()

        with patch.dict(os.environ, {
            "NATS_URL": "nats://x:4222",
            "REDIS_URL": "redis://x:6379",
            "QDRANT_URL": "http://x:6333",
        }):
            async def run():
                with patch(
                    'core.discovery.mesh_discovery.ServiceWatcher'
                ) as MockWatcher:
                    mock_watcher = MagicMock()
                    mock_watcher.get_urls = MagicMock(return_value=ServiceURLs())
                    mock_watcher.start = AsyncMock()
                    MockWatcher.return_value = mock_watcher

                    urls, watcher = await resolve_service_urls(
                        zc=mock_zc, cfg=cfg, timeout=0.1
                    )
                    self.assertIs(watcher, mock_watcher)

            asyncio.run(run())


from config import apply_discovered_urls, MyconexConfig


class TestApplyDiscoveredUrls(unittest.TestCase):

    def _make_cfg(self):
        return MyconexConfig()  # default values

    def test_applies_discovered_urls_when_no_env_var(self):
        """mDNS URLs applied when no env var is set."""
        cfg = self._make_cfg()
        urls = ServiceURLs(
            nats_url="nats://hub:4222",
            redis_url="redis://hub:6379",
            qdrant_url="http://hub:6333",
        )
        env = {k: v for k, v in os.environ.items()
               if k not in ("NATS_URL", "REDIS_URL", "QDRANT_URL")}

        with patch.dict(os.environ, env, clear=True):
            apply_discovered_urls(cfg, urls)

        self.assertEqual(cfg.mesh.nats_url, "nats://hub:4222")
        self.assertEqual(cfg.mesh.redis_url, "redis://hub:6379")
        self.assertEqual(cfg.mesh.qdrant_url, "http://hub:6333")

    def test_does_not_overwrite_explicit_env_var(self):
        """Explicit env var wins — apply_discovered_urls does not overwrite."""
        cfg = self._make_cfg()
        original_nats = cfg.mesh.nats_url  # save the default
        urls = ServiceURLs(
            nats_url="nats://hub:4222",
            redis_url="redis://hub:6379",
            qdrant_url="http://hub:6333",
        )
        with patch.dict(os.environ, {"NATS_URL": "nats://myserver:4222"}):
            apply_discovered_urls(cfg, urls)

        # NATS should NOT be overwritten — user explicitly configured it
        self.assertEqual(cfg.mesh.nats_url, original_nats)
        # Redis and Qdrant should be applied (no env var for them)
        self.assertEqual(cfg.mesh.redis_url, "redis://hub:6379")

    def test_none_urls_not_applied(self):
        """None fields in ServiceURLs are skipped."""
        cfg = self._make_cfg()
        original_nats = cfg.mesh.nats_url
        urls = ServiceURLs(nats_url=None, redis_url="redis://hub:6379")
        env = {k: v for k, v in os.environ.items()
               if k not in ("NATS_URL", "REDIS_URL", "QDRANT_URL")}

        with patch.dict(os.environ, env, clear=True):
            apply_discovered_urls(cfg, urls)

        self.assertEqual(cfg.mesh.nats_url, original_nats)  # unchanged
        self.assertEqual(cfg.mesh.redis_url, "redis://hub:6379")


if __name__ == '__main__':
    unittest.main()
