# Service Discovery — Design Spec

**Date:** 2026-03-26
**Status:** Approved
**Area:** `core/discovery/mesh_discovery.py`, `__main__.py`, `config.py`

---

## Problem

Worker nodes currently require manual IP configuration in `.env` to connect to hub services (NATS, Redis, Qdrant). This breaks the zero-config mesh model and requires updating every worker if the hub's IP changes.

---

## Goal

Worker nodes automatically discover hub services (NATS, Redis, Qdrant) via mDNS on the local network. No manual IP configuration required. Falls back to `.env` if discovery fails. Hard fails with a clear error if neither works.

---

## Constraints

- LAN-only (mDNS does not cross routers — acceptable for target use case)
- No new dependencies — Zeroconf is already used by `MeshDiscovery`
- `.env` configuration must continue to work unchanged (additive, not breaking)
- Hub designation is zero-config — any node probes its own ports and advertises what it finds

---

## Approach

Extend `core/discovery/mesh_discovery.py` with three new additions alongside the existing `MeshDiscovery` class. Use the same Zeroconf stack already in place.

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

### `ServiceAdvertiser`

Runs on any node that has infrastructure services running locally. Probes its own ports at startup and registers mDNS records for each responding service. Unregisters on shutdown.

```python
class ServiceAdvertiser:
    async def start() -> None      # probe ports concurrently, register found services
    async def stop() -> None       # unregister all records, close zeroconf
    async def _probe_port(port: int, timeout: float = 1.0) -> bool  # TCP connect test
```

- Probes ports 4222 (NATS), 6379 (Redis), 6333 (Qdrant) concurrently
- Registers `_nats._tcp.local.`, `_redis._tcp.local.`, `_qdrant._tcp.local.`
- Port probe failures are silent — not having a service is normal on worker nodes
- Shares the `AsyncZeroconf` instance with `MeshDiscovery` to avoid duplicate stacks

### `ServiceWatcher`

Runs on all nodes. Browses mDNS for the three service types. Maintains live `ServiceURLs`. Fires callbacks when any URL changes (hub moved, new IP).

```python
class ServiceWatcher:
    async def start() -> None
    async def stop() -> None
    def get_urls() -> ServiceURLs              # current snapshot
    def on_change(callback: Callable) -> None  # register reconnect callback
```

- Browses `_nats._tcp.local.`, `_redis._tcp.local.`, `_qdrant._tcp.local.` simultaneously
- On add/update/remove: updates `ServiceURLs`, fires all registered callbacks
- Callbacks receive the updated `ServiceURLs` so clients can reconnect

### `ServiceDiscoveryError`

```python
class ServiceDiscoveryError(RuntimeError):
    pass
```

Raised with a human-readable message listing every attempted source:

```
ServiceDiscoveryError: Could not resolve NATS.
  Tried: mDNS (_nats._tcp.local.) — not found after 10s
  Tried: NATS_URL env var — not set
  Fix: start the hub first, or set NATS_URL in .env
```

### `resolve_service_urls()` (public API)

The only function most code needs to call. Handles the full discovery → fallback → error flow.

```python
async def resolve_service_urls(timeout: float = 10.0) -> ServiceURLs
```

---

## Data Flow

### Boot sequence (every node)

```
1. resolve_service_urls(timeout=10s)
   ├── Start ServiceWatcher, browse all 3 service types
   ├── Wait up to 10s for mDNS results
   ├── Merge: discovered URLs override .env defaults
   └── Any URL still None → ServiceDiscoveryError (fail fast, clear message)

2. If local services are running:
   └── ServiceAdvertiser.start()
       ├── Probe ports concurrently (1s timeout each)
       └── Register mDNS record for each responding port

3. Boot continues with resolved URLs
   └── NATS, Redis, Qdrant clients connect normally
```

### Hub IP change (live reconnect)

```
ServiceWatcher detects updated mDNS record
    └── Updates ServiceURLs
    └── Fires on_change callbacks
        └── NATSClient.reconnect(new_url)
        └── RedisClient.reconnect(new_url)
```

### Fallback priority

```
mDNS discovery  →  .env value  →  ServiceDiscoveryError
```

Each service resolves independently — NATS might come from mDNS while Redis falls back to `.env`.

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| mDNS timeout | Silent — moves to .env fallback, no log noise |
| .env fallback used | Logged at `INFO` — visible but not alarming |
| Service not found via either source | `ServiceDiscoveryError` with full diagnostic message |
| Port probe failure on `ServiceAdvertiser` | Silent — worker nodes won't have local services |
| Hub goes offline mid-session | `ServiceWatcher` fires `on_change` with `None` URL; client reconnect logic handles |

---

## Integration Points

Only two files outside `mesh_discovery.py` are touched:

**`__main__.py`**
- Call `resolve_service_urls()` early in boot, before any service clients connect
- Pass returned `ServiceURLs` into config

**`config.py`**
- Accept `ServiceURLs` as optional overrides on top of env var defaults
- `ServiceURLs` values take precedence over env vars when present

---

## What Does Not Change

- `MeshDiscovery` (node-to-node discovery) — untouched
- `.env` configuration — still works exactly as before
- `config/mesh_config.yaml` — unchanged
- No new Python dependencies

---

## Testing

- `ServiceAdvertiser`: mock TCP probe; verify correct mDNS records registered
- `ServiceWatcher`: mock Zeroconf browser events; verify `ServiceURLs` updates and callbacks fire
- `resolve_service_urls()`: test all three paths — mDNS found, .env fallback, hard fail
- Integration: start advertiser on one port, verify watcher resolves it within timeout
