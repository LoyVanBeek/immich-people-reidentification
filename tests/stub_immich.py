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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import library

API_KEY = "reid-read-scoped-test-key"

SEARCH_PATH = "/api/search/metadata"
FACES_PATH = "/api/faces"
PEOPLE_PATH = "/api/people"


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
        # asset id -> face detection dicts (assigned or not, the GET /faces shape)
        self.faces: dict[str, list[dict[str, Any]]] = {}
        # people for /api/people endpoint
        self.people: list[dict[str, Any]] = []
        # tags (PUT /api/tags upsert) and asset->tag assignments (PUT /api/tags/assets)
        self.tags: list[dict[str, Any]] = []
        self.tagged_assets: dict[str, list[str]] = {}
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
        return f"http://{address[0]!s}:{int(address[1])}"

    def search_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] == SEARCH_PATH]

    def other_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] != SEARCH_PATH]

    def faces_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] == FACES_PATH]

    def people_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [r for r in self.requests if r.path.split("?")[0] == PEOPLE_PATH]

    def tags_requests(self) -> list[RecordedRequest]:
        with self._lock:
            return [
                r
                for r in self.requests
                if r.path.split("?")[0] in ("/api/tags", "/api/tags/assets")
            ]

    # -- request handling ---------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # keep test output clean

            def do_GET(self) -> None:
                stub._record(self, b"")
                self._serve_get()

            def do_POST(self) -> None:
                body_bytes = self._read_body()
                stub._record(self, body_bytes)
                path = self.path.split("?")[0]
                if path == SEARCH_PATH:
                    if not self._authorized():
                        self._json(401, {"message": "Invalid or missing API key"})
                        return
                    body = json.loads(body_bytes)
                    self._search(body)
                    return
                if path == FACES_PATH:
                    if not self._authorized():
                        self._json(401, {"message": "Invalid or missing API key"})
                        return
                    body = json.loads(body_bytes)
                    self._create_face(body)
                    return
                if path == PEOPLE_PATH:
                    if not self._authorized():
                        self._json(401, {"message": "Invalid or missing API key"})
                        return
                    body = json.loads(body_bytes)
                    self._create_person(body)
                    return
                self._json(404, {"message": "not found"})

            def do_PUT(self) -> None:
                body_bytes = self._read_body()
                stub._record(self, body_bytes)
                path = self.path.split("?")[0]
                if not self._authorized():
                    self._json(401, {"message": "Invalid or missing API key"})
                    return
                if path == "/api/tags":
                    self._upsert_tags(json.loads(body_bytes))
                    return
                if path == "/api/tags/assets":
                    self._tag_assets(json.loads(body_bytes))
                    return
                self._json(404, {"message": "not found"})

            def _serve_get(self) -> None:
                if not self._authorized():
                    self._json(401, {"message": "Invalid or missing API key"})
                    return
                parsed = urlsplit(self.path)
                if parsed.path == FACES_PATH:
                    self._faces(parsed)
                    return
                if parsed.path == PEOPLE_PATH:
                    self._people(parsed)
                    return
                # Asset info (GET /api/assets/<id>): the JSON metadata.
                info_parts = parsed.path.rstrip("/").split("/")
                if len(info_parts) == 4 and info_parts[2] == "assets":
                    info_asset = stub._asset_for_path(parsed.path)
                    if info_asset is None:
                        self._json(404, {"message": "not found"})
                        return
                    self._json(200, info_asset)
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

            def _faces(self, parsed: Any) -> None:
                asset_id = parse_qs(parsed.query).get("id", [None])[0]
                known = any(
                    a["id"] == asset_id and a["type"] == "IMAGE" for a in stub.assets
                )
                if asset_id is None or not known:
                    self._json(404, {"message": "not found"})
                    return
                faces: list[dict[str, Any]] = stub.faces.get(asset_id, [])
                self._json(200, faces)

            def _people(self, parsed: Any) -> None:
                # Pagination: page and size params (default page=1, size=500)
                query = parse_qs(parsed.query)
                page = int(query.get("page", ["1"])[0])
                size = int(query.get("size", ["500"])[0])
                start = (page - 1) * size
                end = start + size
                window = stub.people[start:end]
                has_next = end < len(stub.people)
                self._json(
                    200,
                    {
                        "people": window,
                        "total": len(stub.people),
                        "hidden": 0,
                        "hasNextPage": has_next,
                    },
                )

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

            def _create_face(self, body: dict[str, Any]) -> None:
                asset_id = body.get("assetId")
                person_id = body.get("personId")
                if not asset_id or not person_id:
                    self._json(400, {"message": "assetId and personId required"})
                    return
                # Check if asset exists
                known = any(
                    a["id"] == asset_id and a["type"] == "IMAGE" for a in stub.assets
                )
                if not known:
                    self._json(404, {"message": "asset not found"})
                    return
                # Check if person exists
                person_known = any(p["id"] == person_id for p in stub.people)
                if not person_known:
                    self._json(404, {"message": "person not found"})
                    return
                # Create face
                face_id = f"face-{asset_id}-{person_id}"
                face = {
                    "id": face_id,
                    "assetId": asset_id,
                    "personId": person_id,
                    "boundingBoxX1": body.get("boundingBoxX1"),
                    "boundingBoxY1": body.get("boundingBoxY1"),
                    "boundingBoxX2": body.get("boundingBoxX2"),
                    "boundingBoxY2": body.get("boundingBoxY2"),
                }
                # Store the created face for potential future GET /faces queries
                if asset_id not in stub.faces:
                    stub.faces[asset_id] = []
                stub.faces[asset_id].append(
                    {
                        "id": face_id,
                        "assetId": asset_id,
                        "person": {
                            "id": person_id,
                            "name": next(
                                (
                                    p["name"]
                                    for p in stub.people
                                    if p["id"] == person_id
                                ),
                                "",
                            ),
                        },
                        "boundingBoxX1": body.get("boundingBoxX1"),
                        "boundingBoxY1": body.get("boundingBoxY1"),
                        "boundingBoxX2": body.get("boundingBoxX2"),
                        "boundingBoxY2": body.get("boundingBoxY2"),
                        "imageWidth": 1440,
                        "imageHeight": 1440,
                        "sourceType": "manual",
                    }
                )
                self._json(200, face)

            def _create_person(self, body: dict[str, Any]) -> None:
                name = body.get("name")
                if not name:
                    self._json(400, {"message": "name required"})
                    return
                # Check if person already exists
                existing = next((p for p in stub.people if p["name"] == name), None)
                if existing:
                    self._json(200, existing)
                    return
                person_id = f"person-{len(stub.people) + 1}"
                person = {"id": person_id, "name": name}
                stub.people.append(person)
                self._json(200, person)

            def _upsert_tags(self, body: dict[str, Any]) -> None:
                names = body.get("tags", [])
                if not isinstance(names, list) or not all(
                    isinstance(n, str) for n in names
                ):
                    self._json(400, {"message": "tags must be a list of strings"})
                    return
                result = []
                for name in names:
                    existing = next((t for t in stub.tags if t["name"] == name), None)
                    if existing:
                        result.append(existing)
                        continue
                    tag = {"id": f"tag-{len(stub.tags) + 1}", "name": name}
                    stub.tags.append(tag)
                    result.append(tag)
                self._json(200, result)

            def _tag_assets(self, body: dict[str, Any]) -> None:
                asset_ids = body.get("assetIds", [])
                tag_ids = body.get("tagIds", [])
                if not isinstance(asset_ids, list) or not isinstance(tag_ids, list):
                    self._json(400, {"message": "assetIds and tagIds required"})
                    return
                for asset_id in asset_ids:
                    stub.tagged_assets.setdefault(asset_id, [])
                    for tag_id in tag_ids:
                        if tag_id not in stub.tagged_assets[asset_id]:
                            stub.tagged_assets[asset_id].append(tag_id)
                self._json(200, {"count": len(asset_ids)})

            def _authorized(self) -> bool:
                return self.headers.get("x-api-key") == API_KEY

            def _read_body(self) -> bytes:
                length = int(self.headers.get("content-length", "0"))
                return self.rfile.read(length)

            def _json(self, status: int, payload: Any) -> None:
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
        if body_bytes and request.headers.get("content-type", "").startswith(
            "application/json"
        ):
            body = json.loads(body_bytes)
        else:
            body = None
        headers = {k.lower(): v for k, v in request.headers.items()}
        with self._lock:
            self.requests.append(
                RecordedRequest(
                    method=request.command,
                    path=request.path,
                    body=body,
                    headers=headers,
                )
            )

    def _asset_for_path(self, path: str) -> dict[str, Any] | None:
        # /api/assets/<id>, /api/assets/<id>/original and /api/assets/<id>/thumbnail
        parts = path.split("?")[0].rstrip("/").split("/")
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "assets":
            for asset in self.assets:
                if asset["id"] == parts[3] and asset["type"] == "IMAGE":
                    return asset
        if (
            len(parts) == 5
            and parts[1] == "api"
            and parts[2] == "assets"
            and parts[4] in ("original", "thumbnail", "preview")
        ):
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
