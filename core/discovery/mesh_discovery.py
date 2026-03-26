"""
MYCONEX Mesh Discovery
mDNS-based peer discovery using python-zeroconf.
Advertises this node as _ai-mesh._tcp and tracks all peers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

logger = logging.getLogger(__name__)

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


SERVICE_TYPE = "_ai-mesh._tcp.local."
SERVICE_VERSION = "0.1.0"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class MeshPeer:
    name: str
    hostname: str
    address: str
    port: int
    tier: str = ""
    roles: list[str] = field(default_factory=list)
    version: str = ""
    last_seen: float = field(default_factory=time.time)
    is_online: bool = True

    @property
    def endpoint(self) -> str:
        return f"http://{self.address}:{self.port}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["endpoint"] = self.endpoint
        return d


# ─── Peer Registry ────────────────────────────────────────────────────────────

class PeerRegistry:
    """Thread-safe in-memory peer store."""

    def __init__(self):
        self._peers: dict[str, MeshPeer] = {}
        self._lock = asyncio.Lock()

    async def add_or_update(self, peer: MeshPeer) -> None:
        async with self._lock:
            existing = self._peers.get(peer.name)
            if existing:
                peer.last_seen = time.time()
                peer.is_online = True
            self._peers[peer.name] = peer
            logger.info(f"[registry] peer registered: {peer.name} ({peer.tier}) @ {peer.address}:{peer.port}")

    async def remove(self, name: str) -> None:
        async with self._lock:
            if name in self._peers:
                self._peers[name].is_online = False
                logger.info(f"[registry] peer offline: {name}")

    async def get_all(self) -> list[MeshPeer]:
        async with self._lock:
            return list(self._peers.values())

    async def get_online(self) -> list[MeshPeer]:
        async with self._lock:
            return [p for p in self._peers.values() if p.is_online]

    async def get_by_tier(self, tier: str) -> list[MeshPeer]:
        async with self._lock:
            return [p for p in self._peers.values() if p.tier == tier and p.is_online]

    async def get_by_role(self, role: str) -> list[MeshPeer]:
        async with self._lock:
            return [p for p in self._peers.values() if role in p.roles and p.is_online]

    def get_sync(self, name: str) -> Optional[MeshPeer]:
        return self._peers.get(name)


# ─── Service Listener ─────────────────────────────────────────────────────────

class MeshServiceListener:
    """Handles add/update/remove events from Zeroconf browser."""

    def __init__(
        self,
        registry: PeerRegistry,
        local_name: str,
        on_peer_join: Optional[Callable[[MeshPeer], None]] = None,
        on_peer_leave: Optional[Callable[[str], None]] = None,
    ):
        self.registry = registry
        self.local_name = local_name
        self.on_peer_join = on_peer_join
        self.on_peer_leave = on_peer_leave
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._handle_add(zc, type_, name), self._get_loop()
        )

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._handle_add(zc, type_, name), self._get_loop()
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._handle_remove(name), self._get_loop()
        )

    async def _handle_add(self, zc: Zeroconf, type_: str, name: str) -> None:
        # Skip self
        short_name = name.replace(f".{type_}", "").replace(type_, "")
        if self.local_name in short_name:
            return

        try:
            info = ServiceInfo(type_, name)
            if not info.request(zc, timeout=3000):
                logger.warning(f"[discovery] timeout resolving {name}")
                return

            addresses = info.parsed_addresses()
            if not addresses:
                return

            address = addresses[0]
            props = {
                k.decode(): v.decode() if isinstance(v, bytes) else v
                for k, v in (info.properties or {}).items()
            }

            roles_raw = props.get("roles", "")
            roles = [r.strip() for r in roles_raw.split(",") if r.strip()]

            peer = MeshPeer(
                name=short_name,
                hostname=info.server or short_name,
                address=address,
                port=info.port or 8765,
                tier=props.get("tier", "T4"),
                roles=roles,
                version=props.get("version", ""),
                last_seen=time.time(),
                is_online=True,
            )

            await self.registry.add_or_update(peer)

            if self.on_peer_join:
                result = self.on_peer_join(peer)
                if asyncio.iscoroutine(result):
                    await result

        except Exception as e:
            logger.error(f"[discovery] error processing {name}: {e}")

    async def _handle_remove(self, name: str) -> None:
        short_name = name.replace(f".{SERVICE_TYPE}", "").replace(SERVICE_TYPE, "")
        await self.registry.remove(short_name)
        if self.on_peer_leave:
            result = self.on_peer_leave(short_name)
            if asyncio.iscoroutine(result):
                await result


# ─── Main Discovery Service ───────────────────────────────────────────────────

class MeshDiscovery:
    """
    Manages mDNS advertisement and peer discovery for MYCONEX.

    Usage:
        discovery = MeshDiscovery(node_name="mynode", tier="T2", roles=["inference"])
        await discovery.start()
        peers = await discovery.registry.get_online()
        await discovery.stop()
    """

    def __init__(
        self,
        node_name: str,
        tier: str,
        roles: list[str],
        api_port: int = 8765,
        on_peer_join: Optional[Callable[[MeshPeer], None]] = None,
        on_peer_leave: Optional[Callable[[str], None]] = None,
    ):
        self.node_name = node_name
        self.tier = tier
        self.roles = roles
        self.api_port = api_port
        self.on_peer_join = on_peer_join
        self.on_peer_leave = on_peer_leave

        self.registry = PeerRegistry()
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._browser: Optional[AsyncServiceBrowser] = None
        self._service_info: Optional[AsyncServiceInfo] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        logger.info(f"[discovery] starting for node={self.node_name} tier={self.tier}")

        self._zeroconf = AsyncZeroconf()

        # Register our service
        await self._register()

        # Browse for peers
        listener = MeshServiceListener(
            registry=self.registry,
            local_name=self.node_name,
            on_peer_join=self.on_peer_join,
            on_peer_leave=self.on_peer_leave,
        )
        listener._loop = asyncio.get_event_loop()

        self._browser = AsyncServiceBrowser(
            self._zeroconf.zeroconf,
            SERVICE_TYPE,
            listener,
        )

        self._running = True
        logger.info(f"[discovery] running — advertising {self.service_name}")

    async def stop(self) -> None:
        if not self._running:
            return

        logger.info("[discovery] stopping...")

        if self._service_info and self._zeroconf:
            await self._zeroconf.async_unregister_service(self._service_info)

        if self._zeroconf:
            await self._zeroconf.async_close()

        self._running = False
        logger.info("[discovery] stopped.")

    async def _register(self) -> None:
        hostname = socket.gethostname()
        addresses = self._get_local_addresses()

        props = {
            b"tier": self.tier.encode(),
            b"roles": ",".join(self.roles).encode(),
            b"version": SERVICE_VERSION.encode(),
            b"node": self.node_name.encode(),
        }

        self._service_info = AsyncServiceInfo(
            SERVICE_TYPE,
            f"{self.service_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(a) for a in addresses],
            port=self.api_port,
            properties=props,
            server=f"{hostname}.local.",
        )

        await self._zeroconf.async_register_service(self._service_info, allow_name_change=True)
        logger.info(f"[discovery] registered: {self.service_name} @ {addresses}:{self.api_port}")

    @property
    def service_name(self) -> str:
        return f"MYCONEX-{self.node_name}"

    @property
    def zeroconf(self) -> "AsyncZeroconf":
        """Return the running AsyncZeroconf instance (raises if not started)."""
        if self._zeroconf is None:
            raise RuntimeError("MeshDiscovery not started — call start() first")
        return self._zeroconf

    def _get_local_addresses(self) -> list[str]:
        addrs = []
        try:
            # Get all non-loopback IPv4 addresses
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = addr[4][0]
                if not ip.startswith("127."):
                    addrs.append(ip)
        except Exception:
            pass

        if not addrs:
            # Fallback: connect to external to find default interface IP
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    addrs.append(s.getsockname()[0])
            except Exception:
                addrs.append("127.0.0.1")

        return list(set(addrs))

    async def get_peers_snapshot(self) -> list[dict]:
        peers = await self.registry.get_online()
        return [p.to_dict() for p in peers]

    async def wait_for_peers(self, min_peers: int = 1, timeout: float = 30.0) -> bool:
        """Block until at least min_peers are discovered or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            peers = await self.registry.get_online()
            if len(peers) >= min_peers:
                return True
            await asyncio.sleep(1.0)
        return False


# ─── CLI / standalone ─────────────────────────────────────────────────────────

async def _demo():
    import argparse

    parser = argparse.ArgumentParser(description="MYCONEX Mesh Discovery Demo")
    parser.add_argument("--name", default=socket.gethostname())
    parser.add_argument("--tier", default="T3")
    parser.add_argument("--roles", default="relay,orchestration")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=int, default=30, help="Seconds to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def on_join(peer: MeshPeer):
        print(f"  [+] Peer joined: {peer.name} ({peer.tier}) @ {peer.address}")

    def on_leave(name: str):
        print(f"  [-] Peer left: {name}")

    discovery = MeshDiscovery(
        node_name=args.name,
        tier=args.tier,
        roles=args.roles.split(","),
        api_port=args.port,
        on_peer_join=on_join,
        on_peer_leave=on_leave,
    )

    await discovery.start()
    print(f"Discovering mesh peers for {args.duration}s... (Ctrl+C to stop)")

    try:
        for _ in range(args.duration):
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        peers = await discovery.get_peers_snapshot()
        print(f"\nDiscovered {len(peers)} peer(s):")
        print(json.dumps(peers, indent=2))
        await discovery.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
