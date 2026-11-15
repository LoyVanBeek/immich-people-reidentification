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
    captured_at: str  # capture time: EXIF when present, else upload time
    original_file_name: str


@dataclass(frozen=True)
class Face:
    """One Immich face detection on an asset, with its person label if any.

    `person_id` is set only when Immich assigned a person (the appearance
    is then already labeled); `person_name` is the person's name and is
    the anchor material when non-empty. Unassigned detections have both
    as None.
    """

    asset_id: str
    left: float
    top: float
    right: float
    bottom: float
    person_id: str | None
    person_name: str | None


@dataclass(frozen=True)
class Person:
    """One Immich person (for autocomplete in the confirmation UI)."""

    id: str
    name: str


@dataclass(frozen=True)
class CreatedPerson:
    """A person created via POST /people."""

    id: str
    name: str


@dataclass(frozen=True)
class CreatedFace:
    """A face created via POST /faces."""

    id: str
    asset_id: str
    person_id: str


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
                        captured_at=_capture_time(item),
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

    def fetch_faces(self, asset_id: str) -> list[Face]:
        """Fetch every face detection on one asset, assigned or not.

        `GET /faces?id=<assetId>` returns all detections including
        unassigned ones (person null) — the anchoring material.
        """
        response = httpx.get(
            f"{self._base_url}/api/faces",
            params={"id": asset_id},
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise ImmichError(response.status_code, _error_message(response))
        body: Any = response.json()
        faces: list[Face] = []
        for item in body:
            person = item.get("person")
            person_id: str | None = None
            person_name: str | None = None
            if isinstance(person, dict):
                raw_id = person.get("id")
                if isinstance(raw_id, str) and raw_id:
                    person_id = raw_id
                raw_name = person.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    person_name = raw_name.strip()
            faces.append(
                Face(
                    asset_id=asset_id,
                    left=float(item["boundingBoxX1"]),
                    top=float(item["boundingBoxY1"]),
                    right=float(item["boundingBoxX2"]),
                    bottom=float(item["boundingBoxY2"]),
                    person_id=person_id,
                    person_name=person_name,
                )
            )
        return faces

    def fetch_people(self) -> list[Person]:
        """Fetch all people from Immich for the name autocomplete.

        Uses `GET /api/people` with pagination (size=1000) to retrieve all people.
        """
        people: list[Person] = []
        page = 1
        while True:
            response = httpx.get(
                f"{self._base_url}/api/people",
                params={"page": page, "size": 1000},
                headers=self._headers,
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                raise ImmichError(response.status_code, _error_message(response))
            body: dict[str, Any] = response.json()
            for item in body.get("people", []):
                person_id = item.get("id")
                person_name = item.get("name")
                if isinstance(person_id, str) and person_id and isinstance(person_name, str) and person_name.strip():
                    people.append(Person(id=person_id, name=person_name.strip()))
            if not body.get("hasNextPage", False):
                break
            page += 1
        return people

    def find_person_by_name(self, name: str) -> Person | None:
        """Find an existing Immich person by exact name match (case-insensitive)."""
        people = self.fetch_people()
        name_lower = name.strip().lower()
        for person in people:
            if person.name.lower() == name_lower:
                return person
        return None

    def create_person(self, name: str) -> CreatedPerson:
        """Create a new person in Immich via POST /people."""
        response = httpx.post(
            f"{self._base_url}/api/people",
            json={"name": name.strip()},
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise ImmichError(response.status_code, _error_message(response))
        body: dict[str, Any] = response.json()
        person_id = body.get("id")
        person_name = body.get("name")
        if not isinstance(person_id, str) or not person_id or not isinstance(person_name, str) or not person_name.strip():
            raise ImmichError(response.status_code, "invalid person creation response")
        return CreatedPerson(id=person_id, name=person_name.strip())

    def create_face(
        self,
        asset_id: str,
        person_id: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> CreatedFace:
        """Create a face entry via POST /faces with a head box bound to a person."""
        response = httpx.post(
            f"{self._base_url}/api/faces",
            json={
                "assetId": asset_id,
                "personId": person_id,
                "boundingBoxX1": left,
                "boundingBoxY1": top,
                "boundingBoxX2": right,
                "boundingBoxY2": bottom,
            },
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            raise ImmichError(response.status_code, _error_message(response))
        body: dict[str, Any] = response.json()
        face_id = body.get("id")
        if not isinstance(face_id, str) or not face_id:
            raise ImmichError(response.status_code, "invalid face creation response")
        return CreatedFace(id=face_id, asset_id=asset_id, person_id=person_id)

    def resolve_head_box(
        self,
        asset_id: str,
        person_box: tuple[float, float, float, float],
        min_containment: float = 0.5,
    ) -> tuple[float, float, float, float] | None:
        """Resolve a head box for the given person box on the asset.

        Prefers Immich's own unassigned face detections that are contained
        in the person box (intersection / face-area >= min_containment).
        Returns the best match's box, or None if no suitable face exists.
        """
        faces = self.fetch_faces(asset_id)
        best_face: Face | None = None
        best_ratio = -1.0
        for face in faces:
            # Only consider unassigned (person_id is None) face detections
            if face.person_id is not None:
                continue
            ratio = self._face_person_overlap(person_box, face)
            if ratio >= min_containment and ratio > best_ratio:
                best_face, best_ratio = face, ratio
        if best_face is not None:
            return (best_face.left, best_face.top, best_face.right, best_face.bottom)
        return None

    @staticmethod
    def _face_person_overlap(
        person_box: tuple[float, float, float, float],
        face: Face,
    ) -> float:
        """How much of the face box sits inside the person box: intersection / face area."""
        left = max(person_box[0], face.left)
        top = max(person_box[1], face.top)
        right = min(person_box[2], face.right)
        bottom = min(person_box[3], face.bottom)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        face_area = max(0.0, face.right - face.left) * max(0.0, face.bottom - face.top)
        return intersection / face_area if face_area > 0.0 else 0.0

    def check_existing_faces(
        self,
        asset_id: str,
        person_box: tuple[float, float, float, float],
        target_person_id: str,
        conflict_containment: float = 0.5,
    ) -> tuple[bool, bool]:
        """Check an asset's existing faces for conflicts and already-labeled status.

        Returns (has_conflict, already_labeled):
        - has_conflict: True if there's a face assigned to a DIFFERENT person
          whose box overlaps the person box (intersection / face-area >= conflict_containment)
        - already_labeled: True if there's a face assigned to the target person
          (regardless of box overlap)
        """
        faces = self.fetch_faces(asset_id)
        has_conflict = False
        already_labeled = False
        for face in faces:
            if face.person_id is None:
                continue
            if face.person_id == target_person_id:
                already_labeled = True
            else:
                ratio = self._face_person_overlap(person_box, face)
                if ratio >= conflict_containment:
                    has_conflict = True
        return has_conflict, already_labeled

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


def _capture_time(item: dict[str, Any]) -> str:
    """The asset's capture time: EXIF when present, upload time otherwise."""
    exif = item.get("exifInfo")
    if isinstance(exif, dict):
        date_time_original = exif.get("dateTimeOriginal")
        if isinstance(date_time_original, str) and date_time_original:
            return date_time_original
    return str(item["createdAt"])


def _error_message(response: httpx.Response) -> str:
    try:
        payload: object = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    message = payload.get("message") if isinstance(payload, dict) else None
    return str(message) if message else response.text[:500] or response.reason_phrase
