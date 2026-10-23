"""Fail-closed network-boundary helpers for fixed-host evidence downloads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import ipaddress
import math
import socket
import time
from typing import Any, Callable, Iterable


class NetworkBoundaryError(ValueError):
    """A network request crossed or could not prove its approved boundary."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.safe_message = message


Resolver = Callable[..., Iterable[tuple[Any, ...]]]
MonotonicClock = Callable[[], float]


def _positive_seconds(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0:
        raise ValueError(f"{field_name} must be finite and strictly positive")
    return selected


def _canonical_global_address(value: Any) -> str:
    text = str(value or "").strip()
    if "%" in text:
        text = text.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(text)
    except ValueError as exc:
        raise NetworkBoundaryError(
            "invalid_network_address",
            "GDELT archive resolution returned an invalid address",
        ) from exc
    if (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise NetworkBoundaryError(
            "non_global_network_address",
            "GDELT archive resolution returned a non-global address",
        )
    return address.compressed


def resolve_global_addresses(
    hostname: str,
    *,
    port: int = 443,
    timeout_seconds: float,
    resolver: Resolver = socket.getaddrinfo,
) -> frozenset[str]:
    """Resolve a fixed host within a deadline and require only global peers."""

    timeout = _positive_seconds(timeout_seconds, field_name="DNS timeout")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paisa-dns")
    def resolve() -> tuple[tuple[Any, ...], ...]:
        return tuple(
            resolver(
                hostname,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        )

    future = executor.submit(resolve)
    try:
        try:
            answers = future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise NetworkBoundaryError(
                "dns_resolution_timeout",
                "GDELT archive DNS resolution exceeded its deadline",
            ) from exc
        except Exception as exc:
            raise NetworkBoundaryError(
                "dns_resolution_error",
                "GDELT archive DNS resolution failed",
            ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    approved: set[str] = set()
    for answer in answers:
        if not isinstance(answer, tuple) or len(answer) < 5:
            raise NetworkBoundaryError(
                "invalid_dns_response",
                "GDELT archive DNS resolution returned an invalid response",
            )
        socket_address = answer[4]
        if not isinstance(socket_address, tuple) or not socket_address:
            raise NetworkBoundaryError(
                "invalid_dns_response",
                "GDELT archive DNS resolution returned an invalid response",
            )
        approved.add(_canonical_global_address(socket_address[0]))
    if not approved:
        raise NetworkBoundaryError(
            "empty_dns_response",
            "GDELT archive DNS resolution returned no addresses",
        )
    return frozenset(approved)


def _socket_peer(response: Any) -> Any:
    explicit = getattr(response, "peer_address", None)
    if explicit is not None:
        return explicit
    raw = getattr(response, "raw", None)
    candidates = (
        getattr(getattr(raw, "_connection", None), "sock", None),
        getattr(getattr(raw, "connection", None), "sock", None),
        getattr(
            getattr(
                getattr(getattr(raw, "_fp", None), "fp", None),
                "raw",
                None,
            ),
            "_sock",
            None,
        ),
    )
    for candidate in candidates:
        getpeername = getattr(candidate, "getpeername", None)
        if callable(getpeername):
            try:
                return getpeername()
            except OSError:
                continue
    raise NetworkBoundaryError(
        "peer_address_unavailable",
        "GDELT archive connection peer could not be verified",
    )


def verify_connected_peer(
    response: Any, *, approved_addresses: frozenset[str]
) -> str:
    """Require the connected peer to be global and in the preflight DNS set."""

    peer = _socket_peer(response)
    raw_address = peer[0] if isinstance(peer, tuple) and peer else peer
    address = _canonical_global_address(raw_address)
    if address not in approved_addresses:
        raise NetworkBoundaryError(
            "dns_peer_mismatch",
            "GDELT archive connection peer differs from approved DNS resolution",
        )
    return address


@dataclass(frozen=True, slots=True)
class MonotonicDeadline:
    """One end-to-end deadline shared by DNS, connect, headers and body reads."""

    expires_at: float
    clock: MonotonicClock = time.monotonic

    @classmethod
    def start(
        cls,
        timeout_seconds: float,
        *,
        clock: MonotonicClock = time.monotonic,
    ) -> "MonotonicDeadline":
        timeout = _positive_seconds(
            timeout_seconds, field_name="Total request timeout"
        )
        return cls(expires_at=clock() + timeout, clock=clock)

    def remaining(self) -> float:
        remaining = self.expires_at - self.clock()
        if remaining <= 0:
            raise NetworkBoundaryError(
                "request_deadline_exceeded",
                "GDELT archive request exceeded its total deadline",
            )
        return remaining

    def check(self) -> None:
        self.remaining()


__all__ = [
    "MonotonicDeadline",
    "NetworkBoundaryError",
    "resolve_global_addresses",
    "verify_connected_peer",
]
