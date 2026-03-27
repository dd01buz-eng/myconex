# Service Discovery — Design Spec

**Date:** 2026-03-26
**Status:** Approved
**Area:** `core/discovery/mesh_discovery.py`, `__main__.py`, `config.py`

---

## Problem

Worker nodes currently require manual IP configuration in `.env` to connect to hub services (NATS, Redis, Qdrant). This breaks the zero-config mesh model and requires updating every worker if the hub's IP changes.

---

## Goal

Worker nodes automatically discover hub services (NATS, Redis, Qdrant) via mDNS on the local network. No manual IP configuration required. Falls back to explicitly user-set `.env` values if discovery fails. Hard fails with a clear error if neither works.

---

## Constraints

- LAN-only (mDNS does not cross routers — acceptable for target use case)
- IPv4 only — matches existing `_get_local_addresses()` behavior in `MeshDiscovery`
- No new Python dependencies — Zeroconf is already present
- `.env` configuration must continue to work unchanged (additive, not breaking)
- Hub designation is zero-config — any node probes its own ports and advertises what it finds

---

## Approach

Extend `core/discovery/mesh_discovery.py` with four new additions. `MeshDiscovery` is minimally modified to expose its `AsyncZeroconf` instance as a property so it can be shared with the new components, avoiding socket conflicts from multiple multicast bindings.

---

## Components

### `ServiceURLs` (dataclass)

Simple value object holding resolved connection strings.

```python
@dataclass
class ServiceURLs:
    nats_url:   str | None = None   # "nats://192.168.1.100:4222"
    redis_url:  str | None = None   # "redis://192.168.1.100:6379"
    qdrant_url: str | None = None   # "http://192.168.1.100:6333"
```

### `ServiceDiscoveryError`

```python
class ServiceDiscoveryError(RuntimeError):
    pass
```

Raised with a human-readable message listing every attempted source:

```
ServiceDiscoveryError: Could not resolve NATS.
  Tried: mDNS (_nats._tcp.local.) — not found after 10s
  Tried: NATS_URL env var — not set by user
  Fix: start the hub first, or set NATS_URL in .env
```

### `ServiceAdvertiser`

Runs on any node that has infrastructure services running locally. Probes its own ports at startup and registers mDNS records for each responding service. Unregisters on shutdown.

```python
class ServiceAdvertiser:
    def __init__(self, zc: AsyncZeroconf, node_name: str, config: MeshConfig)
    async def start() -> None      # probe ports concurrently, register found services
    async def stop() -> None       # unregister all records
    async def _probe_port(port: int, timeout: float = 1.0) -> bool  # TCP connect to 127.0.0.1
```

**Key decisions:**
- Receives `AsyncZeroconf` instance from `MeshDiscovery` (injected, not self-created) to avoid socket conflicts
- Always probes `127.0.0.1` — confirming local service presence only
- Port values parsed from `cfg.mesh.nats_url` / `cfg.mesh.redis_url` / `cfg.mesh.qdrant_url` via `urllib.parse.urlsplit().port`; falls back to well-known defaults (4222 / 6379 / 6333) if the URL contains no explicit port
- Each service mDNS instance name includes `node_name` qualifier: `MYCONEX-{node_name}._nats._tcp.local.` — prevents collision if multiple hubs are on the LAN
- Port probe failures are silent — worker nodes won't have local services
- First-responding hub wins; if multiple hubs advertise the same service type, `ServiceWatcher` uses the first-seen record

### `ServiceWatcher`

Runs on all nodes. Browses mDNS for the three service types. Maintains live `ServiceURLs`. Fires callbacks when any URL changes (hub moved, new IP). Stays alive after initial discovery to support live reconnection.

```python
class ServiceWatcher:
    def __init__(self, zc: AsyncZeroconf)
    async def start() -> None
    async def stop() -> None
    def get_urls() -> ServiceURLs                        # current snapshot
    def on_change(callback: Callable[[ServiceURLs], None]) -> None  # register callback
```

**Callback contract:**
- Signature: `callback(urls: ServiceURLs) -> None`
- Supports both sync and coroutine callbacks — uses the same `asyncio.iscoroutine(result)` check pattern as the existing `MeshServiceListener` (lines 172-174, 183-185 of `mesh_discovery.py`)
- Fires on any change to any individual service URL; always passes the full updated `ServiceURLs` object
- Fired asynchronously via `asyncio.run_coroutine_threadsafe` from Zeroconf's listener thread, matching existing pattern

**Lifecycle:**
- `ServiceWatcher` stays alive after `resolve_service_urls()` returns — it is not torn down after the initial timeout
- The caller (`__main__.py`) holds the reference and calls `stop()` during shutdown
- This allows the same instance to serve both initial discovery and live reconnection callbacks

### `resolve_service_urls()` (public API)

```python
async def resolve_service_urls(
    zc: AsyncZeroconf,
    cfg: MyconexConfig,
    timeout: float = 10.0,
) -> tuple[ServiceURLs, ServiceWatcher]
```

Returns both the resolved `ServiceURLs` and the live `ServiceWatcher` instance. Caller stores the watcher for shutdown and callback registration.

**Priority chain (per service, independently):**

```
1. Explicit user configuration (env var or .env file) — highest priority
2. mDNS discovery (wait up to timeout)
3. ServiceDiscoveryError — if neither source has a value
```

**Explicit config check:** `load_config()` calls `_load_dotenv()` which injects `.env` file values into `os.environ`. By the time `resolve_service_urls()` runs, both shell env vars and `.env` file values are present in `os.environ`. The check `os.environ.get('NATS_URL') is not None` correctly captures both — `.env` file values count as explicit user configuration and take priority over mDNS. This is intentional: if a user has configured a specific server, respect it.

**Hard fail condition:** A service raises `ServiceDiscoveryError` only when `os.environ.get()` returns `None` (no user config, no `.env` value) AND mDNS discovery found nothing within the timeout. The dataclass defaults in `MeshConfig` (`nats://localhost:4222` etc.) are never used as fallback — they are bypassed entirely in this path.

---

## `MeshDiscovery` Change (minimal)

`MeshDiscovery` is modified in one place only — expose the `AsyncZeroconf` instance as a property after `start()`:

```python
@property
def zeroconf(self) -> AsyncZeroconf:
    if self._zeroconf is None:
        raise RuntimeError("MeshDiscovery not started")
    return self._zeroconf
```

No behavioral changes. All existing tests pass unchanged.

---

## `config.py` Integration

Add a standalone function (not a constructor change):

```python
def apply_discovered_urls(cfg: MyconexConfig, urls: ServiceURLs) -> None:
    """Mutate cfg.mesh in-place with mDNS-discovered service URLs.
    Only applies URLs not already set by explicit user configuration.
    """
    if urls.nats_url and os.environ.get("NATS_URL") is None:
        cfg.mesh.nats_url = urls.nats_url
    if urls.redis_url and os.environ.get("REDIS_URL") is None:
        cfg.mesh.redis_url = urls.redis_url
    if urls.qdrant_url and os.environ.get("QDRANT_URL") is None:
        cfg.mesh.qdrant_url = urls.qdrant_url
```

Called in `__main__.py` after `resolve_service_urls()` and before any service clients connect. Correct attribute paths: `cfg.mesh.nats_url`, `cfg.mesh.redis_url`, `cfg.mesh.qdrant_url` (matching lines 134-136 of `config.py`). Explicit user config always wins — `apply_discovered_urls` never overwrites a user-set value.

---

## Data Flow

### Boot sequence (every node)

```
1. load_config() → cfg (from yaml/env/defaults, as before)

2. MeshDiscovery.start() → exposes cfg.zeroconf instance

3. resolve_service_urls(zc=discovery.zeroconf, cfg=cfg, timeout=10s)
   ├── Start ServiceWatcher using shared AsyncZeroconf
   ├── ServiceAdvertiser.start() (probes 127.0.0.1 ports concurrently)
   ├── Wait up to 10s for mDNS results
   ├── Per service: use mDNS result → explicit env var → ServiceDiscoveryError
   └── Returns (ServiceURLs, ServiceWatcher)

4. apply_discovered_urls(cfg, urls)
   └── cfg.mesh.nats_url / cfg.mesh.redis_url / cfg.mesh.qdrant_url updated in-place
       (only for services not already set via explicit user config)

5. Service clients (NATS, Redis, Qdrant) connect using cfg values as normal
```

### Hub IP change (live reconnect)

```
ServiceWatcher detects updated mDNS record
    └── Updates internal ServiceURLs (None for departed services)
    └── Fires on_change callbacks with full updated ServiceURLs
        └── Partial ServiceURLs are valid — some fields may be None
            (services go offline independently, not all-or-nothing)
        └── Clients reconnect to new URL, or handle None gracefully
```

### Priority chain (per service, independently)

```
explicit user config (.env / env var)  →  mDNS discovery  →  ServiceDiscoveryError
```

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| mDNS timeout | Silent — moves to env var check |
| Explicit env var used as fallback | Logged at `INFO` — visible but not alarming |
| Service not found via either source | `ServiceDiscoveryError` with full diagnostic message |
| Port probe failure on `ServiceAdvertiser` | Silent — normal for worker nodes |
| Hub goes offline mid-session | `ServiceWatcher` fires `on_change` with `None` URL; client reconnect logic handles |
| Multiple hubs on LAN | First-seen record wins per service type |

---

## Multi-Hub Disambiguation

If two hub nodes are on the same LAN, each advertises with a unique instance name (`MYCONEX-{node_name}._nats._tcp.local.`). `ServiceWatcher` uses the first-seen record per service type and does not switch unless that record disappears. This is documented behaviour — for controlled environments, naming nodes consistently (e.g. always naming the primary hub `hub`) gives predictable routing.

---

## What Does Not Change

- `MeshDiscovery` node-to-node peer discovery — behavior unchanged; one new property added
- `.env` explicit values — still work as the fallback layer
- `config/mesh_config.yaml` — unchanged
- No new Python dependencies

---

## Testing

**Unit tests (mocked Zeroconf):**
- `ServiceAdvertiser`: mock `_probe_port`, verify correct mDNS records registered per responding port; verify silent behavior on probe failure
- `ServiceWatcher`: inject mock Zeroconf browser events; verify `ServiceURLs` updates correctly; verify both sync and coroutine callbacks fire
- `resolve_service_urls()`: test all three resolution paths — mDNS found, explicit env var fallback, hard fail; verify dataclass defaults do not count as valid fallback

**Integration test note:**
Running `ServiceAdvertiser` and `ServiceWatcher` in the same process requires two `AsyncZeroconf` instances binding to the mDNS multicast socket. Integration tests should mock the Zeroconf layer (inject a fake browser/advertiser) rather than use real multicast, to avoid socket conflicts and make tests deterministic.

**What to verify manually (LAN smoke test):**
1. Hub node starts → `_nats._tcp.local.` visible via `dns-sd -B _nats._tcp` or `avahi-browse`
2. Worker node starts → connects without any `.env` service URLs set
3. Hub IP changes (restart with new IP) → worker reconnects automatically within ~5s
