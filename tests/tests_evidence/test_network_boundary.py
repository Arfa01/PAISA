from __future__ import annotations

import socket

import pytest

from paisa.evidence.network_boundary import (
    MonotonicDeadline,
    NetworkBoundaryError,
    resolve_global_addresses,
    verify_connected_peer,
)


def _resolver(address: str):
    def resolve(*_args):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, 443),
            )
        ]

    return resolve


class _Response:
    def __init__(self, peer_address: str) -> None:
        self.peer_address = (peer_address, 443)


def test_public_resolution_and_matching_peer_are_accepted() -> None:
    approved = resolve_global_addresses(
        "data.gdeltproject.org",
        timeout_seconds=1,
        resolver=_resolver("93.184.216.34"),
    )

    assert approved == frozenset({"93.184.216.34"})
    assert (
        verify_connected_peer(
            _Response("93.184.216.34"), approved_addresses=approved
        )
        == "93.184.216.34"
    )


@pytest.mark.parametrize(
    "address",
    (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "224.0.0.1",
        "192.0.2.1",
    ),
)
def test_non_global_dns_answers_are_rejected(address: str) -> None:
    with pytest.raises(NetworkBoundaryError, match="non-global"):
        resolve_global_addresses(
            "data.gdeltproject.org",
            timeout_seconds=1,
            resolver=_resolver(address),
        )


def test_dns_rebinding_peer_mismatch_is_rejected() -> None:
    approved = resolve_global_addresses(
        "data.gdeltproject.org",
        timeout_seconds=1,
        resolver=_resolver("93.184.216.34"),
    )

    with pytest.raises(NetworkBoundaryError, match="differs"):
        verify_connected_peer(
            _Response("1.1.1.1"), approved_addresses=approved
        )


def test_non_global_connected_peer_is_rejected() -> None:
    with pytest.raises(NetworkBoundaryError, match="non-global"):
        verify_connected_peer(
            _Response("127.0.0.1"),
            approved_addresses=frozenset({"93.184.216.34"}),
        )


def test_monotonic_deadline_expires_fail_closed() -> None:
    now = [10.0]
    deadline = MonotonicDeadline.start(5.0, clock=lambda: now[0])
    assert deadline.remaining() == 5.0

    now[0] = 15.0
    with pytest.raises(NetworkBoundaryError, match="total deadline"):
        deadline.check()
