"""Tests for the two-tab UI API (Proposals tab / Final check tab) — ticket 06.

Covers the ticket-03 evidence payload growth, /api/crop?full=1, and the
ticket-04 endpoints: /api/final-check, /api/write-faces, /api/retarget,
/api/bulk-*, /api/confirm-cluster, plus the cluster_confirmation store table.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from reid.config import load_config
from reid.core import CoreServer
from reid.enums import ProposalStatus
from reid.store import Appearance, Proposal, Store, WriteLedgerEntry
from tests import library
from tests.stub_immich import API_KEY, StubImmich

A1 = library.PHOTO_1_STANDING_ID
A3 = library.PHOTO_3_GROUP_ID
WINDOW = "2026-11-10"
ALICE_ID = "019a5c1e-7c3d-7f0e-9a1b-000000001001"


def _app(
    asset_id: str, appearance: int, box: tuple[float, float, float, float]
) -> Appearance:
    return Appearance(
        asset_id=asset_id,
        appearance=appearance,
        embedding=b"\x00" * 512,
        confidence=0.9,
        box_left=box[0],
        box_top=box[1],
        box_right=box[2],
        box_bottom=box[3],
        pixel_source="preview",
        captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
    )


@pytest.fixture
def ui_stub() -> Iterator[StubImmich]:
    server = StubImmich(library.detection_registry())
    server.people = library.people_registry()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def env(tmp_path: Path, ui_stub: StubImmich) -> Iterator[tuple[CoreServer, Store]]:
    """A core server with a store holding two appearances on one photo (one
    cluster proposed as Alice) and one ask-for-name appearance."""
    weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
    store_path = tmp_path / "twotabs.sqlite3"
    config_path = tmp_path / "twotabs.toml"
    config_path.write_text(
        f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
write_api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{store_path}"

[detection]
pixel_source = "preview"
confidence_floor = 0.35
min_box_height = 100

[head_detection]
weights_path = "{weights_dir / "scrfd_500m.onnx"}"

[clustering]
window_days = 1
distance_threshold = 0.5
min_cluster_size = 2
face_overlap = 0.5

[run]
log_level = "info"
"""
    )
    store = Store(store_path)
    store.add_appearances(
        A1, [_app(A1, 0, (10, 20, 110, 120)), _app(A1, 1, (200, 20, 300, 120))]
    )
    store.add_appearances(A3, [_app(A3, 0, (50, 50, 150, 150))])
    store.replace_proposals(
        [
            Proposal(A1, 0, WINDOW, 0, ProposalStatus.PROPOSED, "Alice"),
            Proposal(A1, 1, WINDOW, 0, ProposalStatus.PROPOSED, "Alice"),
            Proposal(A3, 0, WINDOW, 1, ProposalStatus.ASK_NAME, None),
        ]
    )
    yield CoreServer(load_config(config_path)), store


class TestEvidencePayload:
    """Ticket 03: the evidence card carries everything the new card renders."""

    def test_card_carries_siblings_and_count(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        cards = server.get_evidence({"cluster": ["0"], "window": [WINDOW]})
        assert len(cards) == 2
        card = next(c for c in cards if c["appearance"] == 0)
        # Both stored appearances of the asset, pixel-space boxes, self included.
        assert [s["appearance"] for s in card["siblings"]] == [0, 1]
        assert card["siblings"][0]["box"] == [10, 20, 110, 120]
        assert card["sibling_count"] == 1
        assert card["rejected_name"] is None
        assert card["dismissed"] is False

    def test_card_carries_rejected_name_and_dismissed(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, store = env
        store.add_appearance_rejection(A1, 0, "Alice")
        store.dismiss_cluster(WINDOW, 0, True)
        card = next(
            c
            for c in server.get_evidence({"cluster": ["0"], "window": [WINDOW]})
            if c["appearance"] == 0
        )
        assert card["rejected_name"] == "Alice"
        assert card["dismissed"] is True


class TestCropFull:
    def test_full_serves_whole_photo_at_native_size(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        data, ctype = server.get_crop({"asset_id": [A1], "full": ["1"]})
        assert ctype == "image/jpeg"
        assert data[:2] == b"\xff\xd8"  # JPEG magic
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        # Native pixel-source resolution: the overlay boxes map 1:1 via viewBox.
        assert img.shape[0] >= 1000 and img.shape[1] >= 1000
        # A person crop is much smaller than the full photo.
        crop, _ = server.get_crop({"asset_id": [A1], "appearance": ["0"]})
        assert len(crop) < len(data)

    def test_crop_still_requires_appearance_without_full(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        with pytest.raises(ValueError):
            server.get_crop({"asset_id": [A1]})


class TestFinalCheck:
    def test_persons_grouped_from_proposals_ask_name_excluded(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        payload = server.get_final_check()
        assert [p["name"] for p in payload["persons"]] == ["Alice"]
        alice = payload["persons"][0]
        assert alice["count"] == 2
        assert all(i["source"] == "proposal" for i in alice["items"])
        assert all(i["written"] is False for i in alice["items"])
        # Ticket-03 card fields ride along.
        assert all("siblings" in i and "sibling_count" in i for i in alice["items"])

    def test_written_flag_from_ledger(self, env: tuple[CoreServer, Store]) -> None:
        server, store = env
        store.add_write_ledger_entry(
            WriteLedgerEntry(
                asset_id=A1,
                person_id=ALICE_ID,
                cluster=0,
                window=WINDOW,
                timestamp=datetime.now(UTC),
            )
        )
        payload = server.get_final_check()
        alice = payload["persons"][0]
        written = {i["appearance"]: i["written"] for i in alice["items"]}
        # The write ledger journals asset–person pairs, so both appearances
        # of A1 count as written once (A1, Alice) is in the ledger.
        assert written == {0: True, 1: True}

    def test_pinned_appearance_lands_under_pin_name(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, store = env
        store.set_pin(A1, 1, "Bob")
        payload = server.get_final_check()
        names = {p["name"] for p in payload["persons"]}
        assert names == {"Alice", "Bob"}
        bob = next(p for p in payload["persons"] if p["name"] == "Bob")
        assert bob["items"][0]["source"] == "pin"


class TestWriteFaces:
    def _faces_for(self, ui_stub: StubImmich) -> None:
        # An unassigned Immich face fully inside appearance 0's person box.
        ui_stub.faces[A1] = [
            {
                "id": "face-unassigned-1",
                "assetId": A1,
                "person": None,
                "boundingBoxX1": 15.0,
                "boundingBoxY1": 25.0,
                "boundingBoxX2": 105.0,
                "boundingBoxY2": 115.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

    def test_writes_face_and_journals_ledger(
        self, env: tuple[CoreServer, Store], ui_stub: StubImmich
    ) -> None:
        server, store = env
        self._faces_for(ui_stub)
        result = server.write_faces(
            {"writes": [{"asset_id": A1, "appearance": 0, "name": "Alice"}]}
        )
        assert result["written"] == 1
        assert result["results"][0]["status"] == "written"
        posts = [r for r in ui_stub.faces_requests() if r.method == "POST"]
        assert len(posts) == 1
        assert posts[0].body["assetId"] == A1
        assert posts[0].body["personId"] == ALICE_ID
        assert store.has_write_ledger_entry(A1, ALICE_ID)
        # Idempotent: a second identical write is skipped via the ledger.
        again = server.write_faces(
            {"writes": [{"asset_id": A1, "appearance": 0, "name": "Alice"}]}
        )
        assert again["written"] == 0
        assert again["results"][0]["status"] == "skipped"

    def test_new_person_is_created_on_write(
        self, env: tuple[CoreServer, Store], ui_stub: StubImmich
    ) -> None:
        server, _ = env
        self._faces_for(ui_stub)
        result = server.write_faces(
            {"writes": [{"asset_id": A1, "appearance": 0, "name": "NewPerson"}]}
        )
        assert result["written"] == 1
        people_posts = [r for r in ui_stub.people_requests() if r.method == "POST"]
        assert len(people_posts) == 1
        assert people_posts[0].body["name"] == "NewPerson"

    def test_respects_ignore_and_reject(
        self, env: tuple[CoreServer, Store], ui_stub: StubImmich
    ) -> None:
        server, store = env
        self._faces_for(ui_stub)
        store.ignore_appearance(A1, 1)
        store.add_appearance_rejection(A1, 0, "Alice")
        result = server.write_faces(
            {
                "writes": [
                    {"asset_id": A1, "appearance": 0, "name": "Alice"},
                    {"asset_id": A1, "appearance": 1, "name": "Alice"},
                ]
            }
        )
        assert result["written"] == 0
        assert all(r["status"] == "skipped" for r in result["results"])
        assert not [r for r in ui_stub.faces_requests() if r.method == "POST"]

    def test_unknown_appearance_is_an_error_result(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        result = server.write_faces(
            {"writes": [{"asset_id": A1, "appearance": 99, "name": "Alice"}]}
        )
        assert result["results"][0]["status"] == "error"

    def test_requires_write_scoped_key(
        self, tmp_path: Path, ui_stub: StubImmich, env: tuple[CoreServer, Store]
    ) -> None:
        # A config without write_api_key: nothing can write (dev safety).
        _ = env
        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "nokey.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "nokey.sqlite3"}"

[detection]
pixel_source = "preview"

[head_detection]
weights_path = "{weights_dir / "scrfd_500m.onnx"}"

[clustering]
window_days = 1
distance_threshold = 0.5
min_cluster_size = 2
face_overlap = 0.5
"""
        )
        store2 = Store(tmp_path / "nokey.sqlite3")
        store2.add_appearances(A1, [_app(A1, 0, (10, 20, 110, 120))])
        store2.replace_proposals(
            [Proposal(A1, 0, WINDOW, 0, ProposalStatus.PROPOSED, "Alice")]
        )
        server2 = CoreServer(load_config(config_path))
        result = server2.write_faces(
            {"writes": [{"asset_id": A1, "appearance": 0, "name": "Alice"}]}
        )
        assert "write-scoped API key" in result["error"]


class TestRetargetAndBulk:
    def test_retarget_sets_pins(self, env: tuple[CoreServer, Store]) -> None:
        server, store = env
        result = server.retarget(
            {"items": [{"asset_id": A1, "appearance": 0}], "name": "Bob"}
        )
        assert result["count"] == 1
        assert store.appearance_pins()[(A1, 0)] == "Bob"

    def test_retarget_requires_name(self, env: tuple[CoreServer, Store]) -> None:
        server, _ = env
        assert "error" in server.retarget(
            {"items": [{"asset_id": A1, "appearance": 0}]}
        )

    def test_bulk_ignore_and_unignore(self, env: tuple[CoreServer, Store]) -> None:
        server, store = env
        items = [{"asset_id": A1, "appearance": 0}, {"asset_id": A1, "appearance": 1}]
        result = server.bulk_set_ignore({"items": items}, True)
        assert result["count"] == 2
        assert store.appearance_ignores() == {(A1, 0), (A1, 1)}
        server.bulk_set_ignore({"items": items}, False)
        assert store.appearance_ignores() == set()

    def test_bulk_pin_and_reject(self, env: tuple[CoreServer, Store]) -> None:
        server, store = env
        items = [{"asset_id": A1, "appearance": 0}]
        assert server.bulk_pin({"items": items, "name": "Carol"})["count"] == 1
        assert store.appearance_pins()[(A1, 0)] == "Carol"
        result = server.bulk_reject({"items": items, "name": "Alice"})
        assert result["count"] == 1
        assert (A1, 0, "Alice") in set(store.appearance_rejections())
        prop = next(
            p for p in store.proposals() if p.asset_id == A1 and p.appearance == 0
        )
        assert prop.status == ProposalStatus.REJECTED

    def test_bulk_rejects_unknown_appearances(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        result = server.bulk_set_ignore(
            {"items": [{"asset_id": "nope", "appearance": 0}]}, True
        )
        assert "error" in result


class TestConfirmCluster:
    def test_confirm_marks_cluster_and_pins_name(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, store = env
        result = server.set_cluster_confirmed(
            {"window": WINDOW, "cluster": 0, "name": "Alice", "confirmed": True}
        )
        assert result["confirmed"] is True
        assert (WINDOW, 0) in store.confirmed_clusters()
        # Naming at confirm time pins the cluster's appearances (ask-for-name answer).
        assert store.appearance_pins()[(A1, 0)] == "Alice"
        assert store.appearance_pins()[(A1, 1)] == "Alice"
        state = server.get_ui_state()
        row = next(c for c in state["clusters"] if c["cluster"] == 0)
        assert row["status"] == "confirmed"

    def test_unconfirm_clears(self, env: tuple[CoreServer, Store]) -> None:
        server, store = env
        server.set_cluster_confirmed(
            {"window": WINDOW, "cluster": 0, "confirmed": True}
        )
        server.set_cluster_confirmed(
            {"window": WINDOW, "cluster": 0, "confirmed": False}
        )
        assert store.confirmed_clusters() == set()

    def test_confirm_with_items_pins_only_selected_and_keeps_cluster_open(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        """Confirm proposal with a selection = confirm the selected photos'
        current verdict: only those are pinned (→ Final check), the cluster
        itself stays open for the rest."""
        server, store = env
        result = server.set_cluster_confirmed(
            {
                "window": WINDOW,
                "cluster": 0,
                "name": "Alice",
                "confirmed": True,
                "items": [{"asset_id": A1, "appearance": 0}],
            }
        )
        assert result["confirmed"] is False  # cluster not marked confirmed
        assert result["pinned"] == 1
        assert (WINDOW, 0) not in store.confirmed_clusters()
        assert store.appearance_pins()[(A1, 0)] == "Alice"
        assert (A1, 1) not in store.appearance_pins()  # unselected not pinned
        # The pinned photo surfaces in Final check under Alice.
        final = server.get_final_check()
        alice = next(p for p in final["persons"] if p["name"] == "Alice")
        assert any(i["asset_id"] == A1 and i["appearance"] == 0 for i in alice["items"])

    def test_confirm_with_invalid_items_rejected(
        self, env: tuple[CoreServer, Store]
    ) -> None:
        server, _ = env
        result = server.set_cluster_confirmed(
            {
                "window": WINDOW,
                "cluster": 0,
                "name": "Alice",
                "confirmed": True,
                "items": [{"asset_id": "", "appearance": 0}],
            }
        )
        assert "error" in result

    def test_confirmation_survives_wipe(self, env: tuple[CoreServer, Store]) -> None:
        _, store = env
        store.confirm_cluster(WINDOW, 0, True)
        store.wipe()
        assert (WINDOW, 0) in store.confirmed_clusters()
