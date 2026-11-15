"""The Immich client: one module wrapping the Immich API.

Used by the pipeline (read-scoped key) and, in later tickets, by write-back.
Auth goes through `x-api-key` headers with a fine-grained, read-scoped key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ImmichError(Exception):
    """An Immich API request failed."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"Immich API returned {status}: {message}")


@dataclass(frozen=True)
class Asset:
    """The asset fields the pipeline needs from the checkpointed feed."""

    id: str
    created_at: str
    original_file_name: str


class ImmichClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        page_size: int,
        timeout: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._page_size = page_size
        self._timeout = timeout
        self._headers = {"x-api-key": api_key, "accept": "application/json"}

    def fetch_new_assets(self, created_after: str | None = None) -> list[Asset]:
        """Fetch photo assets created strictly after the given checkpoint.

        Reads every page before returning, so a failure mid-feed surfaces as
        an exception and the caller keeps its previous checkpoint.
        """
        assets: list[Asset] = []
        page = 1
        while True:
            body = self._search_page(created_after, page)
            payload = body["assets"]
            for item in payload["items"]:
                assets.append(
                    Asset(
                        id=item["id"],
                        created_at=item["createdAt"],
                        original_file_name=item.get("originalFileName") or "",
                    )
                )
            next_page = payload.get("nextPage")
            if next_page is None:
                return assets
            page = int(next_page)

    def fetch_pixels(self, asset_id: str, source: str = "preview") -> bytes:
        """Fetch one asset's pixels (JPEG bytes) from Immich.

        The default source is Immich's preview thumbnail (~1440px), which
        the pipeline detects on and crops from; "original" is the fallback
        knob. No thumbnail-to-original coordinate mapping happens in v1.
        """
        path = f"/api/assets/{asset_id}/thumbnail?size=preview"
        if source == "original":
            path = f"/api/assets/{asset_id}/original"
        response = httpx.get(
            f"{self._base_url}{path}",
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise ImmichError(response.status_code, _error_message(response))
        return response.content

    def _search_page(self, created_after: str | None, page: int) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "type": "IMAGE",
            "order": "asc",
            "size": self._page_size,
            "page": page,
        }
        if created_after is not None:
            request_body["createdAfter"] = created_after
        response = httpx.post(
            f"{self._base_url}/api/search/metadata",
            json=request_body,
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise ImmichError(response.status_code, _error_message(response))
        body: dict[str, Any] = response.json()
        return body


def _error_message(response: httpx.Response) -> str:
    try:
        payload: object = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    message = payload.get("message") if isinstance(payload, dict) else None
    return str(message) if message else response.text[:500] or response.reason_phrase
