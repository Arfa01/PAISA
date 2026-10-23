from __future__ import annotations

import signal
import socket
import threading
import time

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


def test_blocked_resolver_is_interrupted_without_a_lingering_worker() -> None:
    before = {thread.ident for thread in threading.enumerate()}

    def blocked_resolver(*_args):
        time.sleep(10)
        return []

    started = time.monotonic()
    with pytest.raises(NetworkBoundaryError) as failure:
        resolve_global_addresses(
            "data.gdeltproject.org",
            timeout_seconds=0.05,
            resolver=blocked_resolver,
        )
    elapsed = time.monotonic() - started

    assert failure.value.error_type == "dns_resolution_timeout"
    assert elapsed < 1.0
    assert {
        thread.ident for thread in threading.enumerate() if thread.ident not in before
    } == set()
    assert not any(
        thread.name.startswith("paisa-dns") for thread in threading.enumerate()
    )


def test_deadline_signal_is_not_swallowed_by_an_ordinary_exception_handler() -> None:
    swallowed = threading.Event()

    def defensive_resolver(*_args):
        try:
            time.sleep(10)
        except Exception:
            swallowed.set()
            time.sleep(10)
        return []

    started = time.monotonic()
    with pytest.raises(NetworkBoundaryError) as failure:
        resolve_global_addresses(
            "data.gdeltproject.org",
            timeout_seconds=0.05,
            resolver=defensive_resolver,
        )

    assert failure.value.error_type == "dns_resolution_timeout"
    assert time.monotonic() - started < 1.0
    assert not swallowed.is_set()


def test_deadline_repeats_when_callback_swallows_first_base_exception() -> None:
    swallowed = threading.Event()

    def defensive_resolver(*_args):
        try:
            time.sleep(10)
        except BaseException:
            swallowed.set()
            time.sleep(10)
        return []

    started = time.monotonic()
    with pytest.raises(NetworkBoundaryError) as failure:
        resolve_global_addresses(
            "data.gdeltproject.org",
            timeout_seconds=0.05,
            resolver=defensive_resolver,
        )

    assert failure.value.error_type == "dns_resolution_timeout"
    assert swallowed.is_set()
    assert time.monotonic() - started < 1.0


def test_resolver_fails_closed_outside_the_main_thread_before_io() -> None:
    called = threading.Event()
    observed: list[NetworkBoundaryError] = []

    def resolver(*_args):
        called.set()
        return []

    def invoke() -> None:
        try:
            resolve_global_addresses(
                "data.gdeltproject.org",
                timeout_seconds=0.05,
                resolver=resolver,
            )
        except NetworkBoundaryError as exc:
            observed.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert not called.is_set()
    assert [failure.error_type for failure in observed] == [
        "interruptible_deadline_unavailable"
    ]


def test_resolver_fails_closed_without_replacing_an_active_process_timer() -> None:
    called = False

    def resolver(*_args):
        nonlocal called
        called = True
        return []

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        with pytest.raises(NetworkBoundaryError) as failure:
            resolve_global_addresses(
                "data.gdeltproject.org",
                timeout_seconds=0.05,
                resolver=resolver,
            )
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    assert failure.value.error_type == "interruptible_deadline_unavailable"
    assert not called
    assert remaining > 0
    assert interval == 0
