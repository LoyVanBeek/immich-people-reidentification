"""UI server: static assets plus a thin proxy to the core API.

This process owns nothing: it serves the browser assets and forwards
every /api/* request to the core API server (reid-core), which is the
single owner of the store. The UI therefore needs no access to the
store file at all.
"""

from __future__ import annotations

import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

from .config import load_config

_log = logging.getLogger("reid.server")

STATIC_DIR = Path(__file__).parent / "ui"

_UI_VERSION: str | None = None


def ui_version() -> str:
    """Stable content hash of the static assets, for cache-busting URLs."""
    global _UI_VERSION
    if _UI_VERSION is None:
        import hashlib

        digest = hashlib.sha256()
        for name in ("app.js", "style.css"):
            digest.update((STATIC_DIR / name).read_bytes())
        _UI_VERSION = digest.hexdigest()[:12]
    return _UI_VERSION


class UIHandler(BaseHTTPRequestHandler):
    """Serves static assets; proxies /api/* to the core."""

    # Class variables set by serve_ui()
    core_url: str = ""
    api_key: str = ""

    def do_GET(self) -> None:
        try:
            self._do_get()
        except Exception as exc:  # noqa: BLE001
            _log.exception("GET %s failed", self.path)
            self._send_text(500, f"internal server error: {exc}")

    def _do_get(self) -> None:
        parsed_path = self.path.split("?")[0]
        if parsed_path == "/" or parsed_path == "/index.html":
            self._serve_index()
        elif parsed_path.startswith("/api/"):
            self._proxy("GET")
        else:
            self._serve_static(parsed_path.lstrip("/"))

    def do_POST(self) -> None:
        try:
            self._proxy("POST")
        except Exception as exc:  # noqa: BLE001
            _log.exception("POST %s failed", self.path)
            self._send_text(500, f"internal server error: {exc}")

    def _proxy(self, method: str) -> None:
        """Forward an /api/* request to the core, auth header injected."""
        core_url = self.core_url.rstrip("/")
        url = f"{core_url}{self.path}"
        length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(length) if length else None
        headers = {"x-api-key": self.api_key}
        if self.headers.get("content-type"):
            headers["content-type"] = self.headers["content-type"]
        try:
            # Generous timeout: confirms can write many faces, crops can
            # involve head detection.
            response = httpx.request(
                method, url, content=body, headers=headers, timeout=900.0
            )
        except httpx.HTTPError as exc:
            _log.error("core unreachable (%s): %s", core_url, exc)
            self._send_text(502, "core is unreachable — is reid-core running?")
            return
        self.send_response(response.status_code)
        content_type = response.headers.get("content-type", "application/json")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)

    def _serve_index(self) -> None:
        """index.html with the asset URLs stamped by a content hash.

        The hash changes whenever app.js or style.css change, so every new
        build gets brand-new asset URLs — a browser can never pair new HTML
        with stale cached JS, which made deploys look like they did nothing.
        """
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("__UI_VERSION__", ui_version())
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_static(self, name: str) -> None:
        import mimetypes

        # Prevent directory traversal
        if ".." in name or name.startswith("/"):
            self.send_error(404)
            return

        path = STATIC_DIR / name
        if not path.is_file():
            self.send_error(404)
            return

        mime, _ = mimetypes.guess_type(name)
        if mime is None:
            mime = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        # Static assets ship with cache-busting versioned URLs; no-store
        # keeps a deploy from looking like it did nothing.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as fh:
            self.wfile.write(fh.read())

    def _send_text(self, status: int, message: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())


def serve_ui(config_path: Path, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the UI server: static assets + proxy to the core."""
    cfg = load_config(config_path)

    UIHandler.core_url = cfg.core.url
    UIHandler.api_key = cfg.core.api_key

    server = HTTPServer((host, port), UIHandler)
    _log.info(
        "UI server listening on http://%s:%d (proxying /api to %s)",
        host,
        port,
        cfg.core.url,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _log.info("UI server stopped")


def main() -> None:
    """Entry point for `reid-ui`."""
    import click

    @click.command()
    @click.option(
        "--config",
        "config_path",
        required=True,
        type=click.Path(path_type=Path, dir_okay=False),
    )
    @click.option("--host", default="0.0.0.0", help="Bind address")
    @click.option("--port", default=8080, type=int, help="Port to listen on")
    def ui(config_path: Path, host: str, port: int) -> None:
        """Run the UI server (static assets + proxy to the core)."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
        serve_ui(config_path, host, port)

    ui()


if __name__ == "__main__":
    main()
