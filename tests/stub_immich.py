"""A stub Immich HTTP server that records every request.

Implements exactly the API slice the pipeline uses so far, with the response
shapes of Immich v2–v3 (`POST /api/search/metadata`, page-number pagination
via the legacy flat fields). The fixture registry is filtered the way the real
server filters: `createdAfter` is strictly greater-than, `type` is exact, and
ordering follows the request's `order` field (default: desc).

Failure injection lets tests simulate a crashed run.
"""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import library

API_KEY = "reid-read-scoped-test-key"

SEARCH_PATH = "/api/search/metadata"


@dataclass
class RecordedRequest:
    method: str
    path: str
    body: Any  # parsed JSON for POSTs with a body, else None
    headers: dict[str, str]


class StubImmich:
    def __init__(self, assets: list[dict[str, Any]]) -> None:
        self.assets = assets
        self.fail_search_pages: set[int] = set()
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    def start(self, port: int = 0) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        address = self._server.server_address
        return f"http://{str(address[0])}:{int(address[1])}"

    def search_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] == SEARCH_PATH]

    def other_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] != SEARCH_PATH]

    # -- request handling ---------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: N802
                pass  # keep test output clean

            def do_GET(self) -> None:  # noqa: N802
                stub._record(self, b"")
                self._serve_get()

            def do_POST(self) -> None:  # noqa: N802
                body_bytes = self._read_body()
                stub._record(self, body_bytes)
                if self.path.split("?")[0] != SEARCH_PATH:
                    self._json(404, {"message": "not found"})
                    return
                if not self._authorized():
                    self._json(401, {"message": "Invalid or missing API key"})
                    return
                body = json.loads(body_bytes)
                self._search(body)

            def _serve_get(self) -> None:
                if not self._authorized():
                    self._json(401, {"message": "Invalid or missing API key"})
                    return
                asset = stub._asset_for_path(self.path)
                if asset is None:
                    self._json(404, {"message": "not found"})
                    return
                payload = library.asset_bytes(asset["id"])
                self.send_response(200)
                self.send_header("content-type", "image/jpeg")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _search(self, body: dict[str, Any]) -> None:
                page = int(body.get("page", 1))
                size = int(body.get("size", 1000))
                if page in stub.fail_search_pages:
                    self._json(500, {"message": "injected failure"})
                    return
                items = stub._filter(body)
                window = items[(page - 1) * size : page * size]
                has_next = page * size < len(items)
                self._json(
                    200,
                    {
                        "albums": {"total": 0, "count": 0, "items": [], "facets": []},
                        "assets": {
                            "total": len(window),
                            "count": len(window),
                            "items": window,
                            "facets": [],
                            "nextPage": str(page + 1) if has_next else None,
                        },
                    },
                )

            def _authorized(self) -> bool:
                return self.headers.get("x-api-key") == API_KEY

            def _read_body(self) -> bytes:
                length = int(self.headers.get("content-length", "0"))
                return self.rfile.read(length)

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                payload_bytes = library.json_bytes(payload)
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)

        return Handler

    # -- helpers ------------------------------------------------------------

    def _record(self, request: BaseHTTPRequestHandler, body_bytes: bytes) -> None:
        body: Any
        if body_bytes and request.headers.get("content-type", "").startswith("application/json"):
            body = json.loads(body_bytes)
        else:
            body = None
        headers = {k.lower(): v for k, v in request.headers.items()}
        with self._lock:
            self.requests.append(
                RecordedRequest(method=request.command, path=request.path, body=body, headers=headers)
            )

    def _asset_for_path(self, path: str) -> dict[str, Any] | None:
        # /api/assets/<id>/original and /api/assets/<id>/thumbnail
        parts = path.split("?")[0].rstrip("/").split("/")
        if len(parts) == 5 and parts[1] == "api" and parts[2] == "assets" and parts[4] in ("original", "thumbnail", "preview"):
            for asset in self.assets:
                if asset["id"] == parts[3] and asset["type"] == "IMAGE":
                    return asset
        return None

    def _filter(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        items = self.assets
        created_after = body.get("createdAfter")
        if created_after is not None:
            items = [a for a in items if a["createdAt"] > created_after]
        asset_type = body.get("type")
        if asset_type is not None:
            items = [a for a in items if a["type"] == asset_type]
        direction = body.get("order", "desc")
        return sorted(items, key=lambda a: a["createdAt"], reverse=direction != "asc")


def main() -> None:
    """Run the stub Immich server from the command line (container smoke test)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    server = StubImmich(library.smoke_registry())
    server.start(port=args.port)
    print(f"stub immich listening on {server.base_url}", flush=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    server.stop()


if __name__ == "__main__":
    main()
