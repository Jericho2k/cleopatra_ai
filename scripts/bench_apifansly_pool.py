"""PERF-006 — measure what a per-call httpx client actually costs.

Spins up a local TLS server with a throwaway self-signed certificate and issues
the same sequence of requests two ways:

* one shared ``httpx.AsyncClient`` (what services/apifansly.py does now);
* a fresh ``httpx.AsyncClient()`` per request, closed afterwards (what it did
  before).

The difference is client construction plus TCP connect plus the TLS handshake.
Over loopback the network legs are ~free, so this is a **lower bound**: on a
real link to the provider each avoided handshake also costs two extra
round trips. Do not quote this number as an internet-level latency saving —
quote it as the floor, and measure the real one against the provider.

Usage:
    python scripts/bench_apifansly_pool.py [--requests 200]
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        body = b'{"data": {"data": {"response": {}}}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


def _self_signed(directory: Path) -> tuple[Path, Path]:
    key = directory / "key.pem"
    cert = directory / "cert.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def _serve(cert: Path, key: Path):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


async def _shared(url: str, count: int, verify) -> float:
    async with httpx.AsyncClient(verify=verify) as client:
        await client.get(url)  # warm the pool; the first call pays the handshake
        started = time.perf_counter()
        for _ in range(count):
            await client.get(url)
        return time.perf_counter() - started


async def _per_call(url: str, count: int, verify) -> float:
    started = time.perf_counter()
    for _ in range(count):
        client = httpx.AsyncClient(verify=verify)
        try:
            await client.get(url)
        finally:
            await client.aclose()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cert, key = _self_signed(directory)
        server, _thread = _serve(cert, key)
        try:
            port = server.server_address[1]
            url = f"https://127.0.0.1:{port}/api/fansly/acct/chats"
            verify = ssl.create_default_context(cafile=str(cert))
            verify.check_hostname = False

            shared = asyncio.run(_shared(url, args.requests, verify))
            per_call = asyncio.run(_per_call(url, args.requests, verify))
        finally:
            server.shutdown()

    results = {
        "requests": args.requests,
        "shared_client_total_s": round(shared, 4),
        "per_call_client_total_s": round(per_call, 4),
        "shared_client_ms_per_request": round(1000 * shared / args.requests, 3),
        "per_call_client_ms_per_request": round(1000 * per_call / args.requests, 3),
        "saved_ms_per_request": round(1000 * (per_call - shared) / args.requests, 3),
        "note": (
            "Loopback TLS. Excludes network round-trip time, so this is the "
            "floor of the real saving, not the whole of it."
        ),
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(f"requests per configuration: {results['requests']}")
        print(
            f"shared pooled client : {results['shared_client_ms_per_request']:.3f} ms/request"
        )
        print(
            f"fresh client per call: {results['per_call_client_ms_per_request']:.3f} ms/request"
        )
        print(f"floor saving         : {results['saved_ms_per_request']:.3f} ms/request")
        print()
        print(results["note"])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(main())
