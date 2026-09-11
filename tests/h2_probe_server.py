"""Real HTTP/2 and HTTP/1.1 origin servers for the PostgREST transport tests.

These exist because the bug they pin is not in our code — it is in how
``httpcore`` handles an HTTP/2 ``GOAWAY``. A mock of our own client cannot
reproduce that, and a test written against one would pass whatever we did to
the transport. So the tests speak real h2 frames over a real socket, and the
production failure string
``<ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>``
comes out of h2 itself rather than out of a fixture.

Cleartext h2 (prior knowledge), so there is no TLS setup and no certificate to
keep alive in CI. ``httpx.Client(http1=False, http2=True)`` is what makes the
client use it; the production client negotiates h2 over ALPN instead, but the
connection-level behaviour under test is identical.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import h2.config
import h2.connection
import h2.events


class _Server:
    """A socket server on a background daemon thread."""

    def __init__(self, handler):
        self._handler = handler
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(32)
        self.port = self._socket.getsockname()[1]
        self.connections = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._socket.accept()
            except OSError:
                return
            self.connections += 1
            index = self.connections
            threading.Thread(
                target=self._safely, args=(conn, index), daemon=True
            ).start()

    def _safely(self, conn, index: int) -> None:
        try:
            self._handler(conn, index)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._socket.close()
        except OSError:
            pass


def _h2_handshake(conn) -> h2.connection.H2Connection:
    state = h2.connection.H2Connection(
        config=h2.config.H2Configuration(client_side=False)
    )
    state.initiate_connection()
    conn.sendall(state.data_to_send())
    return state


def _respond(state, conn, stream_id: int, payload) -> None:
    body = json.dumps(payload).encode()
    state.send_headers(
        stream_id,
        [
            (":status", "200"),
            ("content-type", "application/json"),
            ("content-length", str(len(body))),
        ],
    )
    state.send_data(stream_id, body, end_stream=True)
    conn.sendall(state.data_to_send())


def h2_server_that_goaways_in_flight(*, streams_before_goaway: int) -> _Server:
    """Accept ``streams_before_goaway`` requests, then GOAWAY without answering.

    The GOAWAY names the highest stream it received as ``last_stream_id``, which
    is what makes every in-flight stream fail rather than being re-dispatched:
    httpcore only re-dispatches a stream whose id is ABOVE ``last_stream_id``.
    This is the production case — Supabase's edge recycling a connection that
    has work on it.

    Only the FIRST connection misbehaves. Later ones answer normally, so a test
    can show that a retry on a fresh connection succeeds.
    """

    def handler(conn, index: int) -> None:
        state = _h2_handshake(conn)
        seen: list[int] = []
        while True:
            data = conn.recv(65535)
            if not data:
                return
            for event in state.receive_data(data):
                if not isinstance(event, h2.events.RequestReceived):
                    continue
                seen.append(event.stream_id)
                if index == 1 and len(seen) >= streams_before_goaway:
                    state.close_connection(error_code=0, last_stream_id=max(seen))
                    conn.sendall(state.data_to_send())
                    return
                _respond(state, conn, event.stream_id, [{"connection": index}])
            conn.sendall(state.data_to_send())

    return _Server(handler)


def h2_server_that_holds_until(gate: threading.Event) -> _Server:
    """Hold every request open until ``gate`` is set, then GOAWAY all of them.

    Lets a test put N concurrent requests on ONE multiplexed connection and
    terminate them together, which is the blast radius the production incident
    actually had.
    """

    def handler(conn, index: int) -> None:
        state = _h2_handshake(conn)
        seen: list[int] = []
        if index > 1:
            while True:
                data = conn.recv(65535)
                if not data:
                    return
                for event in state.receive_data(data):
                    if isinstance(event, h2.events.RequestReceived):
                        _respond(state, conn, event.stream_id, [{"connection": index}])
                conn.sendall(state.data_to_send())

        conn.settimeout(0.2)
        while not gate.is_set():
            try:
                data = conn.recv(65535)
            except socket.timeout:
                continue
            if not data:
                return
            for event in state.receive_data(data):
                if isinstance(event, h2.events.RequestReceived):
                    seen.append(event.stream_id)
            conn.sendall(state.data_to_send())
        state.close_connection(error_code=0, last_stream_id=max(seen or [0]))
        conn.sendall(state.data_to_send())

    return _Server(handler)


def http11_server_that_recycles_connections() -> _Server:
    """Answer one request per connection, then close it while it sits idle.

    This is the same peer behaviour as the GOAWAY case — a proxy recycling
    connections — expressed in HTTP/1.1, where httpcore checks whether an idle
    socket has become readable before reusing it.
    """

    def handler(conn, index: int) -> None:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = conn.recv(65535)
            if not chunk:
                return
            buffer += chunk
        body = json.dumps([{"connection": index}]).encode()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: keep-alive\r\n\r\n" + body
        )
        # Let the client see the response and go idle, then recycle.
        time.sleep(0.05)

    return _Server(handler)
