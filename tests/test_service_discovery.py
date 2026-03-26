"""Tests for MYCONEX service discovery (mDNS hub service auto-discovery)."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import ServiceURLs, ServiceDiscoveryError, MeshDiscovery


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


if __name__ == '__main__':
    unittest.main()
