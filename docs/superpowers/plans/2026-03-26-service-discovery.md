# Service Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable MYCONEX worker nodes to automatically discover hub services (NATS, Redis, Qdrant) via mDNS, with fallback to explicit `.env` config and a clear error if neither works.

**Architecture:** Extend `core/discovery/mesh_discovery.py` with `ServiceAdvertiser` (probes own ports, registers mDNS records), `ServiceWatcher` (browses mDNS, fires reconnect callbacks), and `resolve_service_urls()` (discovery → env fallback → hard fail). `MeshDiscovery` gains a `.zeroconf` property to share its `AsyncZeroconf` instance. `config.py` gets `apply_discovered_urls()`. `__main__.py` calls discovery early in boot, before any service clients connect.

**Tech Stack:** Python 3.10+, `zeroconf` (already installed), `urllib.parse` (stdlib), `unittest` + `unittest.mock`

**Spec:** `docs/superpowers/specs/2026-03-26-service-discovery-design.md`

---

## File Map

| File | Change | Responsibility |
|------|--------|---------------|
| `core/discovery/mesh_discovery.py` | Modify | Add `ServiceURLs`, `ServiceDiscoveryError`, `ServiceAdvertiser`, `ServiceWatcher`, `resolve_service_urls()`, `MeshDiscovery.zeroconf` property |
| `config.py` | Modify | Add `apply_discovered_urls()` |
| `__main__.py` | Modify | Add `_discover_services()` helper; call it in boot sequence |
| `tests/test_service_discovery.py` | Create | All unit tests for new components |

---

## Task 1: ServiceURLs dataclass and ServiceDiscoveryError

**Files:**
- Modify: `core/discovery/mesh_discovery.py` (append after line 22, after the existing imports)
- Test: `tests/test_service_discovery.py` (create)

- [ ] **Step 1.1: Create the test file with the first failing test**

Create `tests/test_service_discovery.py`:

```python
"""Tests for MYCONEX service discovery (mDNS hub service auto-discovery)."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.discovery.mesh_discovery import ServiceURLs, ServiceDiscoveryError


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


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd D:/GitHub/myconex
python -m pytest tests/test_service_discovery.py -v
```

Expected: `ImportError: cannot import name 'ServiceURLs'`

- [ ] **Step 1.3: Add ServiceURLs and ServiceDiscoveryError to mesh_discovery.py**

After the existing imports block (after line 22, before `SERVICE_TYPE = ...`), first add `import os` to the standard library imports block (alongside `import socket`, `import time` etc.), then add:

```python
# ─── Service Discovery Types ───────────────────────────────────────────────────

@dataclass
class ServiceURLs:
    """Resolved connection URLs for hub infrastructure services."""
    nats_url:   Optional[str] = None   # e.g. "nats://192.168.1.100:4222"
    redis_url:  Optional[str] = None   # e.g. "redis://192.168.1.100:6379"
    qdrant_url: Optional[str] = None   # e.g. "http://192.168.1.100:6333"


class ServiceDiscoveryError(RuntimeError):
    """Raised when a required service cannot be resolved via mDNS or env vars."""
    pass
```

- [ ] **Step 1.4: Run test to confirm it passes**

```bash
python -m pytest tests/test_service_discovery.py -v
```

Expected: `3 passed`

- [ ] **Step 1.5: Commit**

```bash
git add core/discovery/mesh_discovery.py tests/test_service_discovery.py
git commit -m "feat: add ServiceURLs dataclass and ServiceDiscoveryError"
```

---

## Task 2: MeshDiscovery.zeroconf property

**Files:**
- Modify: `core/discovery/mesh_discovery.py` — add property to `MeshDiscovery` class
- Test: `tests/test_service_discovery.py`

- [ ] **Step 2.1: Write the failing test**

Add to `tests/test_service_discovery.py`:

```python
from core.discovery.mesh_discovery import MeshDiscovery


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
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
python -m pytest tests/test_service_discovery.py::TestMeshDiscoveryZeroconfProperty -v
```

Expected: `AttributeError: 'MeshDiscovery' object has no attribute 'zeroconf'`

- [ ] **Step 2.3: Add the property to MeshDiscovery**

Inside the `MeshDiscovery` class in `core/discovery/mesh_discovery.py`, after the `service_name` property (around line 292), add:

```python
@property
def zeroconf(self) -> "AsyncZeroconf":
    """Return the running AsyncZeroconf instance (raises if not started)."""
    if self._zeroconf is None:
        raise RuntimeError("MeshDiscovery not started — call start() first")
    return self._zeroconf
```

- [ ] **Step 2.4: Run test to confirm it passes**

```bash
python -m pytest tests/test_service_discovery.py -v
```

Expected: all tests pass

- [ ] **Step 2.5: Confirm existing mesh discovery tests still pass**

```bash
python -m pytest tests/test_mesh_discovery.py -v
```

Expected: all pass (no regressions)

- [ ] **Step 2.6: Commit**

```bash
git add core/discovery/mesh_discovery.py tests/test_service_discovery.py
git commit -m "feat: expose MeshDiscovery.zeroconf property for instance sharing"
```

---

## Task 3: ServiceAdvertiser

**Files:**
- Modify: `core/discovery/mesh_discovery.py` — new class after `PeerRegistry`
- Test: `tests/test_service_discovery.py`

The advertiser probes `127.0.0.1` on the hub's service ports and registers mDNS records for each responding service. It also needs a small port-parsing helper.

- [ ] **Step 3.1: Write the failing tests**

Add to `tests/test_service_discovery.py`:

```python
from core.discovery.mesh_discovery import ServiceAdvertiser


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
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_service_discovery.py::TestServiceAdvertiser -v
```

Expected: `ImportError: cannot import name 'ServiceAdvertiser'`

- [ ] **Step 3.3: Implement ServiceAdvertiser**

Add a port-parsing helper and the `ServiceAdvertiser` class to `core/discovery/mesh_discovery.py`, after the `PeerRegistry` class and before `MeshServiceListener`:

```python
# ─── Port Parsing Helper ───────────────────────────────────────────────────────

def _parse_port(url: str, default: int) -> int:
    """Extract port from a URL string; return default if absent or unparseable."""
    from urllib.parse import urlsplit
    try:
        port = urlsplit(url).port
        return port if port is not None else default
    except Exception:
        return default


# ─── Service Advertiser ────────────────────────────────────────────────────────

_SERVICE_PORTS = {
    "_nats._tcp.local.":   4222,
    "_redis._tcp.local.":  6379,
    "_qdrant._tcp.local.": 6333,
}

_SERVICE_URL_ATTRS = {
    "_nats._tcp.local.":   "nats_url",
    "_redis._tcp.local.":  "redis_url",
    "_qdrant._tcp.local.": "qdrant_url",
}


class ServiceAdvertiser:
    """
    Probes localhost ports and registers mDNS records for running hub services.

    Usage:
        advertiser = ServiceAdvertiser(zc=discovery.zeroconf, node_name="hub", cfg=cfg)
        await advertiser.start()
        # ... node runs ...
        await advertiser.stop()
    """

    def __init__(self, zc: "AsyncZeroconf", node_name: str, cfg: Any) -> None:
        self._zc = zc
        self._node_name = node_name
        self._cfg = cfg
        self._registered: list["AsyncServiceInfo"] = []

    async def start(self) -> None:
        """Probe all service ports concurrently; register mDNS for each found."""
        nats_port   = _parse_port(self._cfg.mesh.nats_url,   4222)
        redis_port  = _parse_port(self._cfg.mesh.redis_url,  6379)
        qdrant_port = _parse_port(self._cfg.mesh.qdrant_url, 6333)

        service_map = {
            "_nats._tcp.local.":   nats_port,
            "_redis._tcp.local.":  redis_port,
            "_qdrant._tcp.local.": qdrant_port,
        }

        # Probe all ports concurrently
        results = await asyncio.gather(
            *[self._probe_port(port) for port in service_map.values()],
            return_exceptions=True,
        )

        for (svc_type, port), result in zip(service_map.items(), results):
            if result is True:
                await self._register(svc_type, port)

    async def stop(self) -> None:
        """Unregister all advertised services."""
        for info in self._registered:
            try:
                await self._zc.async_unregister_service(info)
            except Exception:
                pass
        self._registered.clear()

    async def _probe_port(self, port: int, timeout: float = 1.0) -> bool:
        """Return True if something is listening on 127.0.0.1:port."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port),
                timeout=timeout,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            return False

    async def _register(self, svc_type: str, port: int) -> None:
        instance_name = f"MYCONEX-{self._node_name}.{svc_type}"
        addresses = [socket.inet_aton("127.0.0.1")]
        try:
            # Use our actual LAN IP for the advertisement
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
            addresses = [socket.inet_aton(lan_ip)]
        except Exception:
            pass

        info = AsyncServiceInfo(
            svc_type,
            instance_name,
            addresses=addresses,
            port=port,
            properties={
                b"node": self._node_name.encode(),
                b"version": SERVICE_VERSION.encode(),
            },
            server=f"{socket.gethostname()}.local.",
        )
        await self._zc.async_register_service(info, allow_name_change=True)
        self._registered.append(info)
        logger.info(f"[advertiser] registered {svc_type} on port {port}")
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_service_discovery.py::TestServiceAdvertiser -v
```

Expected: `3 passed`

- [ ] **Step 3.5: Run full test suite to check for regressions**

```bash
python -m pytest tests/test_mesh_discovery.py tests/test_service_discovery.py -v
```

Expected: all pass

- [ ] **Step 3.6: Commit**

```bash
git add core/discovery/mesh_discovery.py tests/test_service_discovery.py
git commit -m "feat: add ServiceAdvertiser — probes and advertises hub services via mDNS"
```

---

## Task 4: ServiceWatcher

**Files:**
- Modify: `core/discovery/mesh_discovery.py` — new class
- Test: `tests/test_service_discovery.py`

The watcher browses all three service types and fires callbacks when URLs change. It stays alive after initial resolution.

- [ ] **Step 4.1: Write the failing tests**

Add to `tests/test_service_discovery.py`:

```python
from core.discovery.mesh_discovery import ServiceWatcher


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
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_service_discovery.py::TestServiceWatcher -v
```

Expected: `ImportError: cannot import name 'ServiceWatcher'`

- [ ] **Step 4.3: Implement ServiceWatcher**

Add after `ServiceAdvertiser` in `core/discovery/mesh_discovery.py`:

```python
# ─── Service Watcher ──────────────────────────────────────────────────────────

_SVC_TYPE_TO_ATTR = {
    "_nats._tcp.local.":   "nats_url",
    "_redis._tcp.local.":  "redis_url",
    "_qdrant._tcp.local.": "qdrant_url",
}

_SVC_TYPE_SCHEMES = {
    "_nats._tcp.local.":   "nats",
    "_redis._tcp.local.":  "redis",
    "_qdrant._tcp.local.": "http",
}


class _ServiceBrowseListener:
    """Internal Zeroconf listener that feeds discovered records into ServiceWatcher."""

    def __init__(self, watcher: "ServiceWatcher") -> None:
        self._watcher = watcher
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the event loop stored at start() time (from async context)."""
        return self._loop  # always set by ServiceWatcher.start() before browsers run

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._handle_add(zc, type_, name), self._get_loop()
        )

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._handle_add(zc, type_, name), self._get_loop()
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        attr = _SVC_TYPE_TO_ATTR.get(type_)
        if attr:
            self._watcher._update_url(attr, None)

    async def _handle_add(self, zc: Zeroconf, type_: str, name: str) -> None:
        attr = _SVC_TYPE_TO_ATTR.get(type_)
        if not attr:
            return
        try:
            info = ServiceInfo(type_, name)
            if not info.request(zc, timeout=3000):
                return
            addresses = info.parsed_addresses()
            if not addresses:
                return
            scheme = _SVC_TYPE_SCHEMES.get(type_, "http")
            url = f"{scheme}://{addresses[0]}:{info.port}"
            self._watcher._update_url(attr, url)
        except Exception as e:
            logger.debug(f"[watcher] error resolving {name}: {e}")


class ServiceWatcher:
    """
    Browses mDNS for hub services and maintains live ServiceURLs.
    Stays alive after resolve_service_urls() for continuous reconnect support.

    Usage:
        watcher = ServiceWatcher(zc=discovery.zeroconf)
        await watcher.start()
        watcher.on_change(lambda urls: reconnect(urls))
        urls = watcher.get_urls()
        # ... later ...
        await watcher.stop()
    """

    def __init__(self, zc: "AsyncZeroconf") -> None:
        self._zc = zc
        self._urls = ServiceURLs()
        self._callbacks: list[Callable] = []
        self._browsers: list = []

    async def start(self) -> None:
        """Begin browsing all three service types."""
        listener = _ServiceBrowseListener(self)
        listener._loop = asyncio.get_running_loop()  # capture loop from async context
        for svc_type in _SVC_TYPE_TO_ATTR:
            browser = AsyncServiceBrowser(
                self._zc.zeroconf, svc_type, listener
            )
            self._browsers.append(browser)
        logger.info("[watcher] service discovery started")

    async def stop(self) -> None:
        """Stop all browsers."""
        self._browsers.clear()
        logger.info("[watcher] service discovery stopped")

    def get_urls(self) -> ServiceURLs:
        """Return a snapshot of currently resolved service URLs."""
        return ServiceURLs(
            nats_url=self._urls.nats_url,
            redis_url=self._urls.redis_url,
            qdrant_url=self._urls.qdrant_url,
        )

    def on_change(self, callback: Callable[["ServiceURLs"], None]) -> None:
        """Register a callback (sync or coroutine) fired on any URL change."""
        self._callbacks.append(callback)

    def _update_url(self, attr: str, value: Optional[str]) -> None:
        """Update a URL field and fire all registered callbacks."""
        setattr(self._urls, attr, value)
        snapshot = self.get_urls()
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        for cb in self._callbacks:
            result = cb(snapshot)
            if asyncio.iscoroutine(result):
                if loop:
                    asyncio.ensure_future(result, loop=loop)
                else:
                    asyncio.run(result)
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_service_discovery.py::TestServiceWatcher -v
```

Expected: `5 passed`

- [ ] **Step 4.5: Run full test suite**

```bash
python -m pytest tests/test_mesh_discovery.py tests/test_service_discovery.py -v
```

Expected: all pass

- [ ] **Step 4.6: Commit**

```bash
git add core/discovery/mesh_discovery.py tests/test_service_discovery.py
git commit -m "feat: add ServiceWatcher — browses mDNS for hub services, fires reconnect callbacks"
```

---

## Task 5: resolve_service_urls()

**Files:**
- Modify: `core/discovery/mesh_discovery.py` — new top-level async function
- Test: `tests/test_service_discovery.py`

This is the public API. Priority: explicit env var → mDNS → `ServiceDiscoveryError`.

- [ ] **Step 5.1: Write the failing tests**

Add to `tests/test_service_discovery.py`:

```python
from core.discovery.mesh_discovery import resolve_service_urls


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
                    mock_watcher = AsyncMock()
                    mock_watcher.get_urls.return_value = ServiceURLs()  # empty — mDNS found nothing
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
                    mock_watcher = AsyncMock()
                    mock_watcher.get_urls.return_value = ServiceURLs(
                        nats_url="nats://hub:4222",
                        redis_url="redis://hub:6379",
                        qdrant_url="http://hub:6333",
                    )
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
                    mock_watcher = AsyncMock()
                    mock_watcher.get_urls.return_value = ServiceURLs()  # nothing found
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
                    mock_watcher = AsyncMock()
                    mock_watcher.get_urls.return_value = ServiceURLs()
                    mock_watcher.start = AsyncMock()
                    MockWatcher.return_value = mock_watcher

                    urls, watcher = await resolve_service_urls(
                        zc=mock_zc, cfg=cfg, timeout=0.1
                    )
                    self.assertIs(watcher, mock_watcher)

            asyncio.run(run())
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_service_discovery.py::TestResolveServiceUrls -v
```

Expected: `ImportError: cannot import name 'resolve_service_urls'`

- [ ] **Step 5.3: Implement resolve_service_urls()**

Add at the bottom of `core/discovery/mesh_discovery.py` (before the `_demo` function):

```python
# ─── Public API ───────────────────────────────────────────────────────────────

async def resolve_service_urls(
    zc: "AsyncZeroconf",
    cfg: Any,
    timeout: float = 10.0,
) -> tuple[ServiceURLs, ServiceWatcher]:
    """
    Resolve hub service URLs via mDNS discovery with env var priority.

    Priority per service (independently):
      1. Explicit user config: env var or .env file value (highest priority)
      2. mDNS discovery (waits up to timeout seconds)
      3. ServiceDiscoveryError (fail fast with diagnostic message)

    Returns:
        (ServiceURLs, ServiceWatcher) — URLs and the live watcher (keep for reconnect)

    Raises:
        ServiceDiscoveryError — if any service has no URL from either source
    """
    watcher = ServiceWatcher(zc=zc)
    await watcher.start()

    # Wait for mDNS results
    await asyncio.sleep(min(timeout, 10.0))
    discovered = watcher.get_urls()

    def _resolve(env_key: str, mdns_url: Optional[str], label: str) -> str:
        # Priority 1: explicit user config (env var or .env, both land in os.environ)
        if (v := os.environ.get(env_key)):
            logger.info(f"[discovery] {label}: using explicit config ({env_key}={v})")
            return v
        # Priority 2: mDNS
        if mdns_url:
            logger.info(f"[discovery] {label}: resolved via mDNS → {mdns_url}")
            return mdns_url
        # Priority 3: fail
        raise ServiceDiscoveryError(
            f"Could not resolve {label}.\n"
            f"  Tried: mDNS (_{label.lower()}._tcp.local.) — not found after {timeout:.0f}s\n"
            f"  Tried: {env_key} env var — not set\n"
            f"  Fix: start the hub first, or set {env_key} in .env"
        )

    nats_url   = _resolve("NATS_URL",   discovered.nats_url,   "NATS")
    redis_url  = _resolve("REDIS_URL",  discovered.redis_url,  "Redis")
    qdrant_url = _resolve("QDRANT_URL", discovered.qdrant_url, "Qdrant")

    return ServiceURLs(
        nats_url=nats_url,
        redis_url=redis_url,
        qdrant_url=qdrant_url,
    ), watcher
```

- [ ] **Step 5.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_service_discovery.py::TestResolveServiceUrls -v
```

Expected: `4 passed`

- [ ] **Step 5.5: Run full test suite**

```bash
python -m pytest tests/test_mesh_discovery.py tests/test_service_discovery.py -v
```

Expected: all pass

- [ ] **Step 5.6: Commit**

```bash
git add core/discovery/mesh_discovery.py tests/test_service_discovery.py
git commit -m "feat: add resolve_service_urls() — env var priority, mDNS fallback, hard fail"
```

---

## Task 6: apply_discovered_urls() in config.py

**Files:**
- Modify: `config.py` — add one function after `_apply_yaml()`
- Test: `tests/test_service_discovery.py`

- [ ] **Step 6.1: Write the failing test**

Add to `tests/test_service_discovery.py`:

```python
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
        urls = ServiceURLs(
            nats_url="nats://hub:4222",
            redis_url="redis://hub:6379",
            qdrant_url="http://hub:6333",
        )
        with patch.dict(os.environ, {"NATS_URL": "nats://myserver:4222"}):
            apply_discovered_urls(cfg, urls)

        # NATS should NOT be overwritten — user explicitly configured it
        self.assertEqual(cfg.mesh.nats_url, "nats://localhost:4222")  # default unchanged
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
```

- [ ] **Step 6.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_service_discovery.py::TestApplyDiscoveredUrls -v
```

Expected: `ImportError: cannot import name 'apply_discovered_urls'`

- [ ] **Step 6.3: Implement apply_discovered_urls() in config.py**

Add after `_apply_yaml()` (around line 384) in `config.py`:

```python
def apply_discovered_urls(cfg: "MyconexConfig", urls: "ServiceURLs") -> None:
    """
    Apply mDNS-discovered service URLs to cfg.mesh in-place.

    Only applies a URL when:
      - The ServiceURLs field is not None (service was found)
      - No explicit user env var is set for that service (user config wins)
    """
    from core.discovery.mesh_discovery import ServiceURLs  # avoid circular at module level

    if urls.nats_url and not os.environ.get("NATS_URL"):
        cfg.mesh.nats_url = urls.nats_url
    if urls.redis_url and not os.environ.get("REDIS_URL"):
        cfg.mesh.redis_url = urls.redis_url
    if urls.qdrant_url and not os.environ.get("QDRANT_URL"):
        cfg.mesh.qdrant_url = urls.qdrant_url
```

- [ ] **Step 6.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_service_discovery.py::TestApplyDiscoveredUrls -v
```

Expected: `3 passed`

- [ ] **Step 6.5: Run full test suite**

```bash
python -m pytest tests/test_mesh_discovery.py tests/test_service_discovery.py -v
```

Expected: all pass

- [ ] **Step 6.6: Commit**

```bash
git add config.py tests/test_service_discovery.py
git commit -m "feat: add apply_discovered_urls() to config.py"
```

---

## Task 7: Wire into __main__.py

**Files:**
- Modify: `__main__.py` — add `_discover_services()` helper; call it in `boot_rlm_agent()`

No new tests needed here — the integration is covered by the unit tests above and the manual LAN smoke test. This task wires the pieces together.

- [ ] **Step 7.1: Add `_discover_services()` helper to __main__.py**

Add after the `_load_config()` function (around line 116) in `__main__.py`:

```python
async def _discover_services(cfg: Any, verbose: bool = False) -> tuple[Optional[Any], Optional[Any]]:
    """
    Run mDNS service discovery and update cfg.mesh in-place.

    Accepts the cfg instance from boot_rlm_agent so discovered URLs are applied
    to the same config object used by all service clients.

    Returns (watcher, advertiser) — keep both references for shutdown cleanup.
    Returns (None, None) if discovery is unavailable or not needed.
    """
    try:
        from config import apply_discovered_urls
        from core.discovery.mesh_discovery import (
            ServiceAdvertiser,
            resolve_service_urls,
            ServiceDiscoveryError,
        )
        from zeroconf.asyncio import AsyncZeroconf
    except ImportError as e:
        logger.debug(f"Service discovery unavailable: {e}")
        return None, None

    try:
        zc = AsyncZeroconf()

        # Advertise local services (hub nodes will register; worker nodes probe nothing)
        advertiser = ServiceAdvertiser(zc=zc, node_name=cfg.mesh.node_name or "node", cfg=cfg)
        await advertiser.start()

        # Discover hub services
        urls, watcher = await resolve_service_urls(zc=zc, cfg=cfg, timeout=10.0)
        apply_discovered_urls(cfg, urls)

        if verbose:
            _info(f"[discovery] NATS:   {cfg.mesh.nats_url}")
            _info(f"[discovery] Redis:  {cfg.mesh.redis_url}")
            _info(f"[discovery] Qdrant: {cfg.mesh.qdrant_url}")

        return watcher, advertiser

    except ServiceDiscoveryError as e:
        _error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.debug(f"Service discovery skipped: {e}")
        return None, None
```

- [ ] **Step 7.2: Call _discover_services() early in boot_rlm_agent()**

In `boot_rlm_agent()` (around line 127), add the discovery call right after the `.env` load block (after line ~142, before hardware detection):

```python
    # Discover mesh services via mDNS (falls back to .env, fails loudly if neither)
    # Pass cfg so discovered URLs mutate the same instance used by all service clients
    _service_watcher, _service_advertiser = await _discover_services(cfg=cfg, verbose=verbose)
```

And at the bottom of `boot_rlm_agent()`, return both alongside agent and router so they stay alive. Change the return to:

```python
    return agent, router, _service_watcher, _service_advertiser
```

Then update each caller of `boot_rlm_agent()`. Search for all calls:

```bash
grep -n "boot_rlm_agent" __main__.py
```

For each call site (e.g. `agent, router = await boot_rlm_agent(...)`), update to:

```python
agent, router, _watcher, _advertiser = await boot_rlm_agent(...)
```

On shutdown (in whatever cleanup/finally block exists), call:

```python
if _watcher: await _watcher.stop()
if _advertiser: await _advertiser.stop()
```

- [ ] **Step 7.3: Verify syntax on modified files**

```bash
python -c "import ast; ast.parse(open('__main__.py').read()); print('OK')"
python -c "import ast; ast.parse(open('config.py').read()); print('OK')"
python -c "import ast; ast.parse(open('core/discovery/mesh_discovery.py').read()); print('OK')"
```

Expected: `OK` for all three

- [ ] **Step 7.4: Run full test suite**

```bash
python -m pytest tests/test_mesh_discovery.py tests/test_service_discovery.py -v
```

Expected: all pass

- [ ] **Step 7.5: Commit**

```bash
git add __main__.py
git commit -m "feat: wire mDNS service discovery into boot sequence"
```

---

## Task 8: Final verification and push

- [ ] **Step 8.1: Run the entire test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: no failures related to new code

- [ ] **Step 8.2: Verify the new exports are importable**

```bash
python -c "
from core.discovery.mesh_discovery import (
    ServiceURLs, ServiceDiscoveryError,
    ServiceAdvertiser, ServiceWatcher,
    resolve_service_urls,
)
from config import apply_discovered_urls
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 8.3: Manual LAN smoke test (optional but recommended)**

On the hub machine:
```bash
# Start infrastructure
cd services && docker compose up -d && cd ..

# Start MYCONEX — should log "[advertiser] registered _nats._tcp.local. on port 4222" etc.
python -m myconex --mode worker
```

On a worker machine (no .env mesh service URLs set):
```bash
# Should discover and connect without any NATS_URL/REDIS_URL/QDRANT_URL in .env
python -m myconex --mode worker
```

Check worker logs for: `[discovery] NATS: resolved via mDNS → nats://192.168.x.x:4222`

- [ ] **Step 8.4: Push to origin**

```bash
git push
```

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ServiceDiscoveryError: Could not resolve NATS` | Hub not running or mDNS blocked | Start hub first, or set `NATS_URL` in `.env` |
| Worker connects to localhost instead of hub | `NATS_URL=nats://localhost:4222` in `.env` | Comment out mesh URLs in `.env` to use auto-discovery |
| mDNS not finding hub | Different subnet / firewall blocking port 5353 | Set URLs manually in `.env` on each worker |
| `asyncio.run_coroutine_threadsafe` warning | Async callback on closed loop | Ensure `ServiceWatcher.stop()` called before event loop teardown |
