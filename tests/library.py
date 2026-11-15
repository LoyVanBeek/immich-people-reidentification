"""Fixture asset registry and photo library for container-boundary tests.

The stub Immich server serves these assets. Timestamps are fixed, distinct,
ISO-8601 with the same format Immich returns, so plain string comparison on
the `createdAfter` filter is faithful to chronological comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHOTOS_DIR = Path(__file__).parent / "fixtures" / "photos"

PHOTO_1_STANDING_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000001"
PHOTO_2_BLURRED_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000002"
PHOTO_3_GROUP_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000003"
VIDEO_1_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000004"

PHOTO_1_CREATED_AT = "2026-11-10T08:00:00.000Z"
PHOTO_2_CREATED_AT = "2026-11-10T09:30:00.000Z"
PHOTO_3_CREATED_AT = "2026-11-10T10:15:00.000Z"
VIDEO_1_CREATED_AT = "2026-11-10T09:45:00.000Z"

# Seven more photos, later the same day, for the crash-and-resume slice.
PHOTO_4_INDOOR_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000005"
PHOTO_5_INDOOR_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000006"
PHOTO_6_INDOOR_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000007"
PHOTO_7_INDOOR_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000008"
PHOTO_8_PARK_ID = "019a5c1e-7c3d-7f0e-9a1b-000000000009"
PHOTO_9_PARK_ID = "019a5c1e-7c3d-7f0e-9a1b-00000000000a"
PHOTO_10_PARK_ID = "019a5c1e-7c3d-7f0e-9a1b-00000000000b"

PHOTO_4_CREATED_AT = "2026-11-10T11:00:00.000Z"
PHOTO_5_CREATED_AT = "2026-11-10T11:30:00.000Z"
PHOTO_6_CREATED_AT = "2026-11-10T12:00:00.000Z"
PHOTO_7_CREATED_AT = "2026-11-10T12:30:00.000Z"
PHOTO_8_CREATED_AT = "2026-11-10T13:00:00.000Z"
PHOTO_9_CREATED_AT = "2026-11-10T13:30:00.000Z"
PHOTO_10_CREATED_AT = "2026-11-10T14:00:00.000Z"

# The latest createdAt in the seven-photo registry; the checkpoint after a
# full run over it.
EXTENDED_PHOTO_MAX_CREATED_AT = PHOTO_7_CREATED_AT

# The latest createdAt among the photos; the checkpoint after a full run.
PHOTO_MAX_CREATED_AT = PHOTO_3_CREATED_AT

PHOTO_FILE_NAMES = {
    PHOTO_1_STANDING_ID: "reid-fixture-standing.jpg",
    PHOTO_2_BLURRED_ID: "reid-fixture-blurred-face.jpg",
    PHOTO_3_GROUP_ID: "reid-fixture-group.jpg",
    PHOTO_4_INDOOR_ID: "reid-fixture-indoor-a.jpg",
    PHOTO_5_INDOOR_ID: "reid-fixture-indoor-b.jpg",
    PHOTO_6_INDOOR_ID: "reid-fixture-indoor-c.jpg",
    PHOTO_7_INDOOR_ID: "reid-fixture-indoor-d.jpg",
    PHOTO_8_PARK_ID: "reid-fixture-park-a.jpg",
    PHOTO_9_PARK_ID: "reid-fixture-park-b.jpg",
    PHOTO_10_PARK_ID: "reid-fixture-park-c.jpg",
}


def _asset(asset_id: str, *, created_at: str, asset_type: str) -> dict[str, Any]:
    file_name = PHOTO_FILE_NAMES.get(asset_id, "reid-fixture-clip.mp4")
    return {
        "id": asset_id,
        "type": asset_type,
        "createdAt": created_at,
        "updatedAt": created_at,
        "fileCreatedAt": created_at,
        "fileModifiedAt": created_at,
        "originalFileName": file_name,
        "originalPath": f"/library/{asset_id}/{file_name}",
        "previewPath": f"/library/{asset_id}/preview.jpg",
        "thumbnailPath": f"/library/{asset_id}/thumb.jpg",
        "isArchived": False,
        "isFavorite": False,
        "isTrashed": False,
        "isOffline": False,
        "isEncoded": False,
        "isMotion": False,
        "isReadOnly": False,
        "isExternal": False,
        "visibility": "timeline",
        "duration": "0:00:00.000000" if asset_type == "VIDEO" else None,
        "thumbhash": "4oSCAQHO91t0mH9lmIdHR5V2",
        "libraryId": "019a5c1e-7c3d-7f0e-9a1b-00000000beef",
    }


def asset_registry() -> list[dict[str, Any]]:
    """Three photos of known people plus one video asset (videos are out of scope)."""
    return [
        _asset(PHOTO_1_STANDING_ID, created_at=PHOTO_1_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_2_BLURRED_ID, created_at=PHOTO_2_CREATED_AT, asset_type="IMAGE"),
        _asset(VIDEO_1_ID, created_at=VIDEO_1_CREATED_AT, asset_type="VIDEO"),
        _asset(PHOTO_3_GROUP_ID, created_at=PHOTO_3_CREATED_AT, asset_type="IMAGE"),
    ]


def extended_asset_registry() -> list[dict[str, Any]]:
    """Seven photos, enough for multi-page runs with a small page size."""
    return [
        _asset(PHOTO_1_STANDING_ID, created_at=PHOTO_1_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_2_BLURRED_ID, created_at=PHOTO_2_CREATED_AT, asset_type="IMAGE"),
        _asset(VIDEO_1_ID, created_at=VIDEO_1_CREATED_AT, asset_type="VIDEO"),
        _asset(PHOTO_3_GROUP_ID, created_at=PHOTO_3_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_4_INDOOR_ID, created_at=PHOTO_4_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_5_INDOOR_ID, created_at=PHOTO_5_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_6_INDOOR_ID, created_at=PHOTO_6_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_7_INDOOR_ID, created_at=PHOTO_7_CREATED_AT, asset_type="IMAGE"),
    ]


def late_photos() -> list[dict[str, Any]]:
    """Photos that appear only after a first batch has already checkpointed."""
    return [
        _asset(PHOTO_8_PARK_ID, created_at=PHOTO_8_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_9_PARK_ID, created_at=PHOTO_9_CREATED_AT, asset_type="IMAGE"),
        _asset(PHOTO_10_PARK_ID, created_at=PHOTO_10_CREATED_AT, asset_type="IMAGE"),
    ]


def smoke_registry() -> list[dict[str, Any]]:
    """Every photo exactly once plus the video, for the container smoke test."""
    seen = {a["id"] for a in asset_registry()}
    combined = list(asset_registry())
    for asset in [*extended_asset_registry(), *late_photos()]:
        if asset["id"] not in seen:
            seen.add(asset["id"])
            combined.append(asset)
    return combined


def photo_ids() -> list[str]:
    return [PHOTO_1_STANDING_ID, PHOTO_2_BLURRED_ID, PHOTO_3_GROUP_ID]


def asset_bytes(asset_id: str) -> bytes:
    """The fixture image bytes for one asset, read from the photo library."""
    file_name = PHOTO_FILE_NAMES[asset_id]
    return (PHOTOS_DIR / file_name).read_bytes()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value).encode()
