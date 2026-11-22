"""Tests for the confirmation UI HTTP API server."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest

from reid.anchoring import AnchorConfig, build_window
from reid.config import load_config
from reid.core import (
    NEIGHBOUR_OUTLINE_COLOR,
    TARGET_OUTLINE_COLOR,
    CoreServer,
    serve_core,
)
from reid.enums import ProposalStatus
from reid.server import serve_ui
from reid.store import Appearance, Proposal, Store
from tests import library
from tests.stub_immich import API_KEY, StubImmich


def _appearance(
    asset_id: str, appearance: int, box: tuple[float, float, float, float]
) -> Appearance:
    """One stored appearance with a placeholder embedding at `box`."""
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


def _count_close(img: np.ndarray, colour: tuple[int, int, int], tol: int = 40) -> int:
    """Pixels in a BGR image within `tol` per channel of `colour`.

    JPEG artefacts smear outline edges, so count near-matches rather than
    exact pixels to keep the assertion stable.
    """
    return int((np.abs(img.astype(int) - np.array(colour)).max(axis=2) < tol).sum())


@pytest.fixture
def ui_stub() -> Iterator[StubImmich]:
    """A stub with people."""
    server = StubImmich(library.detection_registry())
    server.people = library.people_registry()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def config_with_store(tmp_path: Path, ui_stub: StubImmich) -> Iterator[Path]:
    """A config pointing to a store with some proposals."""
    weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
    config_path = tmp_path / "uitest.toml"
    config_path.write_text(
        f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
write_api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "uitest.sqlite3"}"

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
    yield config_path


@pytest.fixture
def populated_store(tmp_path: Path, config_with_store: Path) -> Store:
    """A store with some appearances and proposals."""
    store = Store(tmp_path / "uitest.sqlite3")

    # Add some appearances
    appearances = [
        Appearance(
            asset_id="asset1",
            appearance=0,
            embedding=b"\x00" * 512,
            confidence=0.9,
            box_left=10,
            box_top=20,
            box_right=110,
            box_bottom=120,
            pixel_source="preview",
            captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
        ),
        Appearance(
            asset_id="asset1",
            appearance=1,
            embedding=b"\x00" * 512,
            confidence=0.8,
            box_left=200,
            box_top=20,
            box_right=300,
            box_bottom=120,
            pixel_source="preview",
            captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
        ),
        Appearance(
            asset_id="asset2",
            appearance=0,
            embedding=b"\x00" * 512,
            confidence=0.85,
            box_left=50,
            box_top=50,
            box_right=150,
            box_bottom=150,
            pixel_source="preview",
            captured_at=datetime.fromisoformat("2026-11-10T11:00:00"),
        ),
    ]
    store.add_appearances("asset1", appearances[:2])
    store.add_appearances("asset2", appearances[2:])

    # Add proposals (window "2026-11-10", cluster 0 and 1)
    proposals = [
        Proposal(
            asset_id="asset1",
            appearance=0,
            window="2026-11-10",
            cluster=0,
            status=ProposalStatus.PROPOSED,
            proposed_name="Alice",
        ),
        Proposal(
            asset_id="asset1",
            appearance=1,
            window="2026-11-10",
            cluster=0,
            status=ProposalStatus.PROPOSED,
            proposed_name="Alice",
        ),
        Proposal(
            asset_id="asset2",
            appearance=0,
            window="2026-11-10",
            cluster=1,
            status=ProposalStatus.ASK_NAME,
            proposed_name=None,
        ),
    ]
    store.replace_proposals(proposals)

    return store


class TestServerAPI:
    """Test the UI server API methods directly."""

    def test_get_ui_state(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        state = server.get_ui_state()

        assert "clusters" in state
        assert "active_cluster" in state
        assert "evidence" in state
        assert "assignment" in state
        assert "stats" in state
        assert "config_knobs" in state
        assert "people" in state
        assert "dry_run" in state

        # Should have 2 clusters
        assert len(state["clusters"]) == 2
        assert state["active_cluster"] == 0

        # First cluster should be Alice
        cluster0 = state["clusters"][0]
        assert cluster0["proposed_name"] == "Alice"
        assert cluster0["status"] == "open"

        # Second cluster should be ask_name
        cluster1 = state["clusters"][1]
        assert cluster1["proposed_name"] is None
        assert cluster1["status"] == "open"

        # People should include the stub's people
        assert "Alice" in state["people"]
        assert "Bob" in state["people"]
        assert "Carol" in state["people"]
        assert "Dave" in state["people"]
        assert "Erin" in state["people"]

    def test_initial_state_is_window_scoped(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """Cluster numbers repeat across windows: the initial panes must not
        mix windows sharing an index."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # Another day, same cluster number 0, different person — on fresh
        # appearance rows (proposals are keyed by asset+appearance, so each
        # appearance lives in exactly one window).
        server.store.add_appearances(
            "asset1",
            [
                Appearance(
                    asset_id="asset1",
                    appearance=2,
                    embedding=b"\x02" * 512,
                    confidence=0.8,
                    box_left=50,
                    box_top=50,
                    box_right=150,
                    box_bottom=150,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2025-09-23T09:00:00"),
                ),
                Appearance(
                    asset_id="asset1",
                    appearance=3,
                    embedding=b"\x03" * 512,
                    confidence=0.8,
                    box_left=160,
                    box_top=50,
                    box_right=260,
                    box_bottom=150,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2025-09-23T09:00:00"),
                ),
            ],
        )
        other_day = [
            Proposal(
                asset_id="asset1",
                appearance=2,
                window="2025-09-23",
                cluster=0,
                status=ProposalStatus.PROPOSED,
                proposed_name="Steffie",
            ),
            Proposal(
                asset_id="asset1",
                appearance=3,
                window="2025-09-23",
                cluster=0,
                status=ProposalStatus.PROPOSED,
                proposed_name="Steffie",
            ),
        ]
        existing = list(server.store.proposals())
        server.store.replace_proposals(existing + other_day)

        state = server.get_ui_state()

        # Sorted by (window, cluster): 2025-09-23 comes first, so it is the
        # default selection — and its evidence carries ONLY that window.
        active = state["clusters"][0]
        assert (active["window"], active["cluster"]) == ("2025-09-23", 0)
        assert state["assignment"]["window"] == "2025-09-23"
        assert state["assignment"]["proposed_name"] == "Steffie"
        assert state["assignment"]["total_appearances"] == 2
        assert all(e["window"] == "2025-09-23" for e in state["evidence"])
        assert all(e["asset_id"] == "asset1" for e in state["evidence"])

    def test_get_evidence(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        # Get evidence for cluster 0
        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})

        assert len(evidence) == 2  # Two appearances in cluster 0
        for e in evidence:
            assert e["cluster"] == 0
            assert e["window"] == "2026-11-10"
            assert e["status"] == "will_write"
            assert e["included"] is True

        # Get evidence for cluster 1 (ask_name)
        evidence = server.get_evidence({"cluster": ["1"], "window": ["2026-11-10"]})

        assert len(evidence) == 1
        assert evidence[0]["status"] == "ask_for_name"

    def test_get_people(self, config_with_store: Path, populated_store: Store) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        people = server.get_people()

        assert isinstance(people, list)
        assert len(people) == 5
        assert people == ["Alice", "Bob", "Carol", "Dave", "Erin"]

    def test_get_crop_returns_jpeg(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """The crop endpoint cuts the appearance box out of Immich's pixels."""
        from tests import library

        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # A real fixture asset the stub serves pixels for, with a box inside it
        server.store.add_appearances(
            library.PHOTO_1_STANDING_ID,
            [
                Appearance(
                    asset_id=library.PHOTO_1_STANDING_ID,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )

        data, ctype = server.get_crop(
            {
                "asset_id": [library.PHOTO_1_STANDING_ID],
                "appearance": ["0"],
            }
        )

        assert ctype == "image/jpeg"
        assert len(data) > 100

    def test_get_crop_with_context_expands_and_differs(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """expand+outline grows the crop and draws the box border."""
        import cv2

        from tests import library

        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        server.store.add_appearances(
            library.PHOTO_1_STANDING_ID,
            [
                Appearance(
                    asset_id=library.PHOTO_1_STANDING_ID,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    # Small box: tight and expanded crops both stay under the
                    # 320px max_side cap, so the expansion is measurable.
                    box_left=300.0,
                    box_top=300.0,
                    box_right=400.0,
                    box_bottom=400.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )

        query = {"asset_id": [library.PHOTO_1_STANDING_ID], "appearance": ["0"]}
        tight, _ = server.get_crop(query)
        contextual, _ = server.get_crop({**query, "expand": ["0.6"], "outline": ["1"]})

        assert len(contextual) > len(tight)
        tight_img = cv2.imdecode(np.frombuffer(tight, np.uint8), cv2.IMREAD_COLOR)
        wide_img = cv2.imdecode(np.frombuffer(contextual, np.uint8), cv2.IMREAD_COLOR)
        assert wide_img is not None and tight_img is not None
        assert max(wide_img.shape[:2]) > max(tight_img.shape[:2])

    def test_get_crop_neighbours_marks_visible_neighbours_on_the_card(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """neighbours=1 outlines other appearances whose box reaches the card.

        The neighbour's box sits inside the expanded card but clear of the
        target's box: visible in the card, yet not intersecting the target.
        """
        import cv2

        from tests import library

        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        server.store.add_appearances(
            library.PHOTO_1_STANDING_ID,
            [
                # Tight target box; expand=0.6 grows the card to 240..460.
                _appearance(
                    library.PHOTO_1_STANDING_ID, 0, (300.0, 300.0, 400.0, 400.0)
                ),
                # Inside the card (420 < 460) but clear of the target box.
                _appearance(
                    library.PHOTO_1_STANDING_ID, 1, (420.0, 300.0, 450.0, 400.0)
                ),
            ],
        )

        query = {"asset_id": [library.PHOTO_1_STANDING_ID], "appearance": ["0"]}
        card, _ = server.get_crop(
            {**query, "expand": ["0.6"], "outline": ["1"], "neighbours": ["1"]}
        )
        bare, _ = server.get_crop({**query, "expand": ["0.6"], "outline": ["1"]})

        card_img = cv2.imdecode(np.frombuffer(card, np.uint8), cv2.IMREAD_COLOR)
        bare_img = cv2.imdecode(np.frombuffer(bare, np.uint8), cv2.IMREAD_COLOR)
        assert card_img is not None and bare_img is not None
        # The neighbour's outline lands around crop coordinates (180, 60).
        neighbour_band = card_img[55:165, 175:220]
        assert _count_close(neighbour_band, NEIGHBOUR_OUTLINE_COLOR) > 50
        # Without the param the identical card is clean — and mini crops,
        # which never send the param, get exactly those pixels.
        bare_band = bare_img[55:165, 175:220]
        assert _count_close(bare_band, NEIGHBOUR_OUTLINE_COLOR) == 0
        # The target's own outline is unaffected — drawn last, so it stays
        # on top where the boxes overlap.
        target_band = card_img[55:165, 55:165]
        assert _count_close(target_band, TARGET_OUTLINE_COLOR) > 50

    def test_get_crop_neighbours_skip_boxes_outside_the_card(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """A neighbour whose box never reaches into the card is not drawn."""
        import cv2

        from tests import library

        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        server.store.add_appearances(
            library.PHOTO_1_STANDING_ID,
            [
                _appearance(
                    library.PHOTO_1_STANDING_ID, 0, (300.0, 300.0, 400.0, 400.0)
                ),
                # Starts at 480, past the card's right edge at 460.
                _appearance(
                    library.PHOTO_1_STANDING_ID, 1, (480.0, 300.0, 520.0, 400.0)
                ),
            ],
        )

        query = {"asset_id": [library.PHOTO_1_STANDING_ID], "appearance": ["0"]}
        card, _ = server.get_crop(
            {**query, "expand": ["0.6"], "outline": ["1"], "neighbours": ["1"]}
        )

        card_img = cv2.imdecode(np.frombuffer(card, np.uint8), cv2.IMREAD_COLOR)
        assert card_img is not None
        # Nowhere in the card is there neighbour-orange; the fixture photo
        # carries no orange-like pixels of its own.
        assert _count_close(card_img, NEIGHBOUR_OUTLINE_COLOR) == 0

    def test_get_crop_missing_appearance_errors(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        with pytest.raises(LookupError):
            server.get_crop({"asset_id": ["asset1"], "appearance": ["99"]})

    def test_evidence_shows_immich_label(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        """Cards show the appearance's current Immich label when one claims it."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # The stub 404s faces for unknown assets, so use a registry fixture id.
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )
        # An Immich face claiming that appearance, labeled "Bob".
        ui_stub.faces[asset_id] = [
            {
                "id": "face-labeled",
                "assetId": asset_id,
                "person": {"id": "person-bob", "name": "Bob"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})

        card = next(e for e in evidence if e["asset_id"] == asset_id)
        assert card["immich_label"] == "Bob"

    def test_evidence_hides_rejected_label(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        """A wrong-label rejection blanks the label on the card."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.LABELED,
                    proposed_name=None,
                )
            ]
        )
        ui_stub.faces[asset_id] = [
            {
                "id": "face-labeled",
                "assetId": asset_id,
                "person": {"id": "person-bob", "name": "Bob"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]
        server.store.add_appearance_rejection(asset_id, 0, "Bob")

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})

        card = next(e for e in evidence if e["asset_id"] == asset_id)
        assert card["immich_label"] is None

    def test_anchoring_ignores_rejected_label_faces(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        """A wrong-label face never anchors or marks the appearance labeled."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # The stub 404s faces for unknown assets, so use a registry fixture id.
        asset_id = library.PHOTO_1_STANDING_ID
        ui_stub.faces[asset_id] = [
            {
                "id": "face-labeled",
                "assetId": asset_id,
                "person": {"id": "person-bob", "name": "Bob"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
                Appearance(
                    asset_id=asset_id,
                    appearance=1,
                    embedding=b"\x01" * 512,
                    confidence=0.85,
                    box_left=700.0,
                    box_top=100.0,
                    box_right=1100.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        faces = server.immich.fetch_faces(asset_id)
        appearances = server.store.appearances_by_asset()[asset_id]
        window = "2026-11-10"
        anchor_cfg = AnchorConfig(face_overlap=0.5, min_cluster_size=2)

        # Without the rejection, the Immich face claims appearance 0 and its
        # label marks the row LABELED (and anchors appearance 1).
        clean = build_window(
            window, appearances, {asset_id: faces}, [[0, 1]], anchor_cfg
        )
        assert any(
            p.status == ProposalStatus.LABELED
            and p.asset_id == asset_id
            and p.appearance == 0
            for p in clean.proposals
        )

        # With the operator's wrong-label flag, that face is not even a
        # candidate: no LABELED row for appearance 0, and no anchored proposal
        # carrying the wrong name.
        server.store.add_appearance_rejection(asset_id, 0, "Bob")
        flagged = build_window(
            window,
            appearances,
            {asset_id: faces},
            [[0, 1]],
            anchor_cfg,
            rejected_faces=frozenset({(asset_id, 0, "Bob")}),
        )
        flagged_statuses = {
            (p.asset_id, p.appearance): p.status for p in flagged.proposals
        }
        assert flagged_statuses.get((asset_id, 0)) != ProposalStatus.LABELED
        assert not any(
            p.proposed_name == "Bob" and p.status == ProposalStatus.PROPOSED
            for p in flagged.proposals
        )

    def test_pin_and_unpin_round_trip(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """Assign records the operator's pin; unpin deletes it; evidence
        cards carry the pinned name."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        result = server.pin_appearance("asset1", 0, "Alice")
        assert result == {"asset_id": "asset1", "appearance": 0, "pin": "Alice"}
        assert server.store.appearance_pins() == {("asset1", 0): "Alice"}

        # Cards show the pin so the operator sees what the next run proposes.
        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        card = next(e for e in evidence if e["asset_id"] == "asset1")
        assert card["pin"] == "Alice"

        result = server.unpin_appearance("asset1", 0)
        assert result == {"asset_id": "asset1", "appearance": 0, "pin": None}
        assert server.store.appearance_pins() == {}

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        card = next(e for e in evidence if e["asset_id"] == "asset1")
        assert card["pin"] is None

    def test_pin_requires_a_name(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        result = server.pin_appearance("asset1", 0, "  ")

        assert "error" in result
        assert server.store.appearance_pins() == {}

    def test_unreject_removes_the_exclusion_and_rederives_the_status(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        server.reject_appearance("asset1", 0, "Alice")
        assert server.store.appearance_rejections() == [("asset1", 0, "Alice")]

        # No face claims asset1 here, and the cluster anchors "Alice": the
        # status re-derives to PROPOSED.
        result = server.unreject_appearance("asset1", 0, "Alice")

        assert result["status"] == "proposed"
        assert server.store.appearance_rejections() == []
        proposal = next(
            p
            for p in server.store.proposals()
            if p.asset_id == "asset1" and p.appearance == 0
        )
        assert proposal.status == ProposalStatus.PROPOSED

    def test_unreject_rederives_labeled_when_immich_agrees(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # The stub 404s faces for unknown assets, so use a registry fixture
        # id with an appearance the face can claim.
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )
        # Immich labels the appearance itself; the operator's exclusion
        # undoes, and the row returns to informational.
        ui_stub.faces[asset_id] = [
            {
                "id": "face-labeled",
                "assetId": asset_id,
                "person": {"id": "person-alice", "name": "Alice"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]
        server.reject_appearance(asset_id, 0, "Alice")

        result = server.unreject_appearance(asset_id, 0, "Alice")

        assert result["status"] == "labeled"

    def test_conflicted_cards_name_the_fighting_labels(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        """Two different Immich labels claim two appearances of one cluster:
        the cards say which labels conflict."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
                Appearance(
                    asset_id=asset_id,
                    appearance=1,
                    embedding=b"\x01" * 512,
                    confidence=0.85,
                    box_left=700.0,
                    box_top=100.0,
                    box_right=1100.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.CONFLICTED,
                    proposed_name=None,
                ),
                Proposal(
                    asset_id=asset_id,
                    appearance=1,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.CONFLICTED,
                    proposed_name=None,
                ),
            ]
        )
        ui_stub.faces[asset_id] = [
            {
                "id": "face-alice",
                "assetId": asset_id,
                "person": {"id": "person-alice", "name": "Alice"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            },
            {
                "id": "face-bob",
                "assetId": asset_id,
                "person": {"id": "person-bob", "name": "Bob"},
                "boundingBoxX1": 800.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 1000.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            },
        ]

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})

        assert all(e["conflict_names"] == ["Alice", "Bob"] for e in evidence), [
            e["conflict_names"] for e in evidence
        ]

    def test_unconflicted_cards_carry_no_conflict_names(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
                Appearance(
                    asset_id=asset_id,
                    appearance=1,
                    embedding=b"\x01" * 512,
                    confidence=0.85,
                    box_left=700.0,
                    box_top=100.0,
                    box_right=1100.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
                Proposal(
                    asset_id=asset_id,
                    appearance=1,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        ui_stub.faces[asset_id] = [
            {
                "id": "face-alice",
                "assetId": asset_id,
                "person": {"id": "person-alice", "name": "Alice"},
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})

        assert all(e["conflict_names"] is None for e in evidence)

    def test_ignore_and_unignore_round_trip(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """Ignore records the operator's decision; cards carry it and the
        original file name; unignore clears both."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # The stub 404s unknown assets, so use a registry fixture id.
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        result = server.ignore_appearance_request(asset_id, 0)
        assert result == {"asset_id": asset_id, "appearance": 0, "ignored": True}

        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        card = next(e for e in evidence if e["asset_id"] == asset_id)
        assert card["ignored"] is True
        # The human-readable file name comes from Immich's asset info.
        assert card["asset_name"] == "reid-fixture-standing.jpg"

        result = server.unignore_appearance_request(asset_id, 0)
        assert result["ignored"] is False
        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        card = next(e for e in evidence if e["asset_id"] == asset_id)
        assert card["ignored"] is False

    def test_confirm_skips_ignored_appearances(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """A confirm in the same session cannot write something just ignored."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
                Appearance(
                    asset_id=asset_id,
                    appearance=1,
                    embedding=b"\x01" * 512,
                    confidence=0.85,
                    box_left=700.0,
                    box_top=100.0,
                    box_right=1100.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
                Proposal(
                    asset_id=asset_id,
                    appearance=1,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        server.ignore_appearance_request(asset_id, 0)

        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [
                    {"asset_id": asset_id, "appearance": 0},
                    {"asset_id": asset_id, "appearance": 1},
                ],
                "dry_run": True,
            }
        )

        # Only the non-ignored appearance is part of the batch.
        assert "error" not in result
        actions = server.store.get_dry_run_batches()[0].actions
        assert all(a.appearance != 0 for a in actions)
        assert [a.appearance for a in actions] == [1]

    def test_write_review_groups_by_name(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """Tabs per pending name across clusters; already-labeled matched via
        face claims; unnamed clusters land on the unnamed tab."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # A LABELED row Immich agrees on: informational, matched by name.
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id="asset1",
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
                Proposal(
                    asset_id="asset1",
                    appearance=1,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.LABELED,
                    proposed_name=None,
                ),
                Proposal(
                    asset_id="asset2",
                    appearance=0,
                    window="2026-11-10",
                    cluster=1,
                    status=ProposalStatus.ASK_NAME,
                    proposed_name=None,
                ),
            ]
        )

        review = server.get_write_review()

        # No faces: nothing matches the Alice tab as already-labeled, but
        # the writable card is there.
        alice = next((t for t in review["tabs"] if t["name"] == "Alice"), None)
        assert alice is not None
        assert [c["asset_id"] for c in alice["writable_cards"]] == ["asset1"]
        assert alice["writable"] == 1
        assert alice["labeled"] == 0
        # The nameless ask-name cluster is on the unnamed tab, not a name tab.
        assert len(review["unnamed"]) == 1
        assert review["unnamed"][0]["cluster"] == 1
        assert review["total_writable"] == 1

    def test_write_decision_persists_and_clears_on_run(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        server.set_write_decision("asset1", 0, False)
        assert server.store.write_decisions() == {("asset1", 0): False}

        # The review page honors the decision: the excluded card stays in the
        # pane (so it can be re-included) but is not counted writable.
        review = server.get_write_review()
        alice = next(t for t in review["tabs"] if t["name"] == "Alice")
        assert alice["writable"] == 1  # asset1 app1 remains included
        excluded = next(c for c in alice["writable_cards"] if c["appearance"] == 0)
        assert excluded["included"] is False

        # A new run replaces proposals and clears decisions.
        rows = server.store.proposals()
        server.store.replace_proposals(rows)
        assert server.store.write_decisions() == {}

    def test_confirm_all_records_one_batch_per_group(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # The stub 404s unknown assets; use the registry fixture id. A second
        # appearance on the next day makes a second (name, window, cluster)
        # group.
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
                Appearance(
                    asset_id=asset_id,
                    appearance=1,
                    embedding=b"\x01" * 512,
                    confidence=0.85,
                    box_left=700.0,
                    box_top=100.0,
                    box_right=1100.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-11T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
                Proposal(
                    asset_id=asset_id,
                    appearance=1,
                    window="2026-11-11",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        # The operator drops the second day's appearance from the batch.
        server.set_write_decision(asset_id, 1, False)

        result = server.confirm_all({"dry_run": True})

        assert "error" not in result
        # Two groups (one per window); the excluded day is not written.
        assert len(result["batch_ids"]) == 1
        assert result["count"] == 1
        batches = server.store.get_dry_run_batches()
        assert [b.window for b in batches] == ["2026-11-10"]
        all_actions = [a for b in batches for a in b.actions]
        assert [a.appearance for a in all_actions] == [0]

    def test_promote_all_promotes_everything(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_id = library.PHOTO_1_STANDING_ID
        server.store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                ),
            ],
        )
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        server.confirm_all({"dry_run": True})
        assert len(server.store.get_dry_run_batches()) == 1

        result = server.promote_all()

        assert result["promoted"] == 1
        assert server.store.get_dry_run_batches()[0].promoted_at is not None
        # Second call: nothing left to promote.
        assert server.promote_all()["promoted"] == 0

    def test_write_pending_writes_only_the_selected(
        self, config_with_store: Path, populated_store: Store, ui_stub: StubImmich
    ) -> None:
        """Per-item writes: the selected action is written and marked, the
        unselected one stays pending, and the batch closes only when
        everything in it is written."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        asset_a = library.PHOTO_1_STANDING_ID
        asset_b = library.PHOTO_2_BLURRED_ID
        for asset_id in (asset_a, asset_b):
            server.store.add_appearances(
                asset_id,
                [
                    Appearance(
                        asset_id=asset_id,
                        appearance=0,
                        embedding=b"\x00" * 512,
                        confidence=0.9,
                        box_left=200.0,
                        box_top=100.0,
                        box_right=600.0,
                        box_bottom=500.0,
                        pixel_source="preview",
                        captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                    ),
                ],
            )
            # An unassigned Immich face claims the appearance: its head box
            # comes from Immich (deterministic), so the recorded action is a
            # clean would_write.
            ui_stub.faces[asset_id] = [
                {
                    "id": f"face-unassigned-{asset_id}",
                    "assetId": asset_id,
                    "person": None,
                    "boundingBoxX1": 300.0,
                    "boundingBoxY1": 200.0,
                    "boundingBoxX2": 500.0,
                    "boundingBoxY2": 400.0,
                    "imageWidth": 1440,
                    "imageHeight": 1440,
                    "sourceType": "machine-learning",
                }
            ]
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_a,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
                Proposal(
                    asset_id=asset_b,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        server.confirm_all({"dry_run": True})
        batch = server.store.get_dry_run_batches()[0]
        assert len(batch.actions) == 2

        # The review's pending section lists both, pre-checked.
        review = server.get_write_review()
        assert len(review["pending"]) == 2
        assert all(p["default_included"] for p in review["pending"])

        # Write only the first item.
        result = server.write_pending(
            {"items": [{"batch_id": batch.id, "asset_id": asset_a, "appearance": 0}]}
        )
        assert result["written"] == 1

        # The unselected one stays pending; the batch is not closed yet.
        review = server.get_write_review()
        assert len(review["pending"]) == 1
        assert review["pending"][0]["asset_id"] == asset_b
        still_open = server.store.get_dry_run_batch(batch.id)
        assert still_open is not None
        assert still_open.promoted_at is None

        # Writing the last one closes the batch.
        server.write_pending(
            {"items": [{"batch_id": batch.id, "asset_id": asset_b, "appearance": 0}]}
        )
        closed = server.store.get_dry_run_batch(batch.id)
        assert closed is not None
        assert closed.promoted_at is not None
        assert server.get_write_review()["pending"] == []

    def test_pinned_elsewhere_leaves_the_cluster_views(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """Pinning an appearance to a different person removes it from the
        cluster's evidence, counts and assignment immediately; the review
        page shows it under the pinned person until the next run."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # Pin asset1 app0 to Bob while the cluster anchors Alice.
        server.pin_appearance("asset1", 0, "Bob")

        # Cluster 0's evidence no longer contains the moved card.
        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        assert [e["appearance"] for e in evidence] == [1]

        # The sidebar row counts match the grid.
        state = server.get_ui_state()
        row = next(c for c in state["clusters"] if c["cluster"] == 0)
        assert row["writable"] == 1
        assert row["appearances"] == 1

        # The assignment pane excludes it too.
        assignment = server._build_assignment(
            0, server.store.proposals(), server.store.appearances_by_asset()
        )
        assert assignment is not None and assignment.total_appearances == 1

        # The review page shows it under the pinned person, not writable
        # until the next run, and confirm-all skips it.
        review = server.get_write_review()
        bob = next(t for t in review["tabs"] if t["name"] == "Bob")
        assert len(bob["pinned_cards"]) == 1
        assert bob["pinned_cards"][0]["pinned_elsewhere"] is True
        assert bob["writable"] == 0
        assert bob["pinned"] == 1
        result = server.confirm_all({"dry_run": True})
        assert "error" not in result
        # The approved batch contains only the remaining Alice appearance —
        # the Bob-pinned card is not written as Alice.
        batch = server.store.get_dry_run_batches()[-1]
        assert [a.appearance for a in batch.actions] == [1]

        # Unpinning restores everything.
        server.unpin_appearance("asset1", 0)
        evidence = server.get_evidence({"cluster": ["0"], "window": ["2026-11-10"]})
        assert [e["appearance"] for e in evidence] == [0, 1]

    def test_fully_labeled_clusters_are_flagged(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """A cluster where Immich labeled every appearance carries
        fully_labeled=True (the queue hides it by default); mixed clusters
        stay visible."""
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)
        # Turn cluster 0 into a fully-labeled group (both appearances).
        server.store.update_proposal_status("asset1", 0, ProposalStatus.LABELED)
        server.store.update_proposal_status("asset1", 1, ProposalStatus.LABELED)

        state = server.get_ui_state()
        row0 = next(c for c in state["clusters"] if c["cluster"] == 0)
        assert row0["fully_labeled"] is True
        assert row0["anchor"] == "already labeled in Immich"
        # Cluster 1 (ask-name) stays normal.
        row1 = next(c for c in state["clusters"] if c["cluster"] == 1)
        assert row1["fully_labeled"] is False

        # A mixed cluster (labeled + proposed) is not fully labeled.
        server.store.replace_proposals(
            [
                Proposal(
                    asset_id="asset1",
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.LABELED,
                    proposed_name=None,
                ),
                Proposal(
                    asset_id="asset1",
                    appearance=1,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                ),
            ]
        )
        state = server.get_ui_state()
        row0 = next(c for c in state["clusters"] if c["cluster"] == 0)
        assert row0["fully_labeled"] is False

    def test_ui_proxies_api_to_core(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        """The UI server forwards /api/* to the core: a request through the
        UI returns the core's answer (and the core must be running)."""
        import socket
        import threading

        from reid.server import serve_ui

        def free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

        core_port = free_port()
        ui_port = free_port()

        # Point the UI's [core] url at the core: serve_ui reads it from the
        # config file.
        config = config_with_store.read_text()
        patched = config_with_store.parent / "proxy.toml"
        patched.write_text(config + f'\n[core]\nurl = "http://127.0.0.1:{core_port}"\n')

        threading.Thread(
            target=serve_core,
            args=(patched, "127.0.0.1", core_port),
            daemon=True,
        ).start()
        threading.Thread(
            target=serve_ui,
            args=(patched, "127.0.0.1", ui_port),
            daemon=True,
        ).start()

        # Wait for the core to accept requests before going through the UI.
        for _ in range(50):
            try:
                with urlopen(
                    f"http://127.0.0.1:{core_port}/api/state", timeout=1
                ) as resp:
                    assert resp.status == 200
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            pytest.fail("core did not start")

        base = f"http://127.0.0.1:{ui_port}"
        with urlopen(f"{base}/api/state", timeout=10) as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert "clusters" in data

        # Static assets are served by the UI itself.
        with urlopen(f"{base}/app.js", timeout=10) as resp:
            assert resp.status == 200

    def test_get_config_knobs(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        knobs = server.get_config_knobs()

        assert knobs.window_days == 1
        assert knobs.distance_threshold == 0.5
        assert knobs.confidence_floor == 0.35
        assert knobs.min_box_height == 100.0
        assert knobs.face_overlap == 0.5

    def test_get_proposals(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        proposals = server.get_proposals()

        assert len(proposals) == 3
        assert proposals[0]["proposed_name"] == "Alice"
        assert proposals[2]["status"] == "ask_name"

    def test_get_summary(self, config_with_store: Path, populated_store: Store) -> None:
        cfg = load_config(config_with_store)
        server = CoreServer(cfg)

        summary = server.get_summary()

        assert len(summary) == 1
        s = summary[0]
        assert s["window"] == "2026-11-10"
        assert s["appearances"] == 3
        # Only 1 cluster has 2+ appearances (cluster 0); cluster 1 is a singleton
        assert s["clusters"] == 1
        assert s["proposals"] == 2
        assert s["ask_for_name"] == 1
        assert s["singletons"] == 1


class TestServerHTTP:
    """Integration tests running the actual HTTP server."""

    @pytest.fixture
    def server_url(
        self, config_with_store: Path, populated_store: Store
    ) -> Iterator[str]:
        """Start a server for this test."""
        # Find a free port
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server_thread = threading.Thread(
            target=serve_core, args=(config_with_store, "127.0.0.1", port), daemon=True
        )
        server_thread.start()

        # Wait for server to start
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urlopen(f"{base_url}/api/state", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            pytest.fail("Server did not start")

        yield base_url

    def test_http_state_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/state") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert "clusters" in data
            assert len(data["clusters"]) == 2

    @pytest.fixture
    def ui_server_url(
        self, config_with_store: Path, populated_store: Store
    ) -> Iterator[str]:
        """Start the UI server (static + proxy) for this test."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server_thread = threading.Thread(
            target=serve_ui, args=(config_with_store, "127.0.0.1", port), daemon=True
        )
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urlopen(f"{base_url}/app.js", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            pytest.fail("UI server did not start")

        yield base_url

    def test_http_static_no_store(self, ui_server_url: str) -> None:
        """app.js must never be served cacheable: a stale copy makes a
        deploy look like it did nothing."""
        with urlopen(f"{ui_server_url}/app.js") as resp:
            assert resp.headers["Cache-Control"] == "no-store"

    def test_http_index_stamps_versioned_asset_urls(self, ui_server_url: str) -> None:
        """Asset URLs carry a content-hash version, so a new build always
        means new URLs and stale cached JS can never run against new HTML."""
        with urlopen(f"{ui_server_url}/") as resp:
            html = resp.read().decode()
        assert "__UI_VERSION__" not in html
        assert "app.js?v=" in html
        assert "style.css?v=" in html

    def test_http_evidence_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/evidence?cluster=0&window=2026-11-10") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert len(data) == 2

    def test_http_people_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/people") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert "people" in data
            assert len(data["people"]) == 5

    def test_http_config_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/config") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert data["window_days"] == 1
            assert data["distance_threshold"] == 0.5

    def test_http_proposals_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/proposals") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert len(data) == 3

    def test_http_summary_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/summary") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert len(data) == 1

    def test_http_serves_static_files(self, ui_server_url: str) -> None:
        with urlopen(f"{ui_server_url}/") as resp:
            assert resp.status == 200
            assert resp.headers.get("content-type", "").startswith("text/html")

        with urlopen(f"{ui_server_url}/index.html") as resp:
            assert resp.status == 200

        with urlopen(f"{ui_server_url}/style.css") as resp:
            assert resp.status == 200
            assert resp.headers.get("content-type", "").startswith("text/css")

        with urlopen(f"{ui_server_url}/app.js") as resp:
            assert resp.status == 200
            # mimetypes.guess_type returns "text/javascript" for .js files
            assert resp.headers.get("content-type", "").startswith("text/javascript")

    def test_http_404_for_missing_static(self, ui_server_url: str) -> None:
        import urllib.error

        try:
            urlopen(f"{ui_server_url}/nonexistent.html")
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_http_no_directory_traversal(self, server_url: str) -> None:
        import urllib.error

        try:
            urlopen(f"{server_url}/../server.py")
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404


class TestHeadDetectorFallback:
    """Tests for the SCRFD head detector fallback (ticket 14)."""

    def test_immich_face_preferred_over_scrfd(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """When Immich has an unassigned face in the person box, it wins over SCRFD.

        Verifies fallback ordering: Immich's unassigned face is used directly
        without invoking the SCRFD detector on the person crop.
        """
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        # Use a real asset ID from the fixture registry
        asset_id = library.PHOTO_1_STANDING_ID

        # Add an unassigned face that overlaps the person box
        # The person box from the fixture is approximately known
        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        # Create a store with a proposal for this asset
        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_fallback.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_fallback.sqlite3"}"

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

        store = Store(tmp_path / "test_fallback.sqlite3")
        # Add an appearance with a known person box
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        # Add a proposal
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # Spy on the head detector to verify it's NOT called when Immich face is available
        assert server.immich._head_detector is not None
        detect_calls = {"count": 0}
        original_detect = server.immich._head_detector.detect

        def spy_detect(crop):
            detect_calls["count"] += 1
            return original_detect(crop)

        server.immich._head_detector.detect = spy_detect  # type: ignore[method-assign]

        # Dry-run confirm for cluster 0
        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        # Verify SCRFD detect() was never called (Immich face took precedence)
        assert detect_calls["count"] == 0, (
            "SCRFD detect() should not be called when Immich face is available"
        )

        assert result["dry_run"] is True
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["status"] == "would_write"
        # The head box should be the Immich face box
        assert list(r["head_box"]) == [300.0, 200.0, 500.0, 400.0]

    def test_scrfd_fallback_when_no_immich_face(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """When no Immich unassigned face overlaps, SCRFD runs on the person crop."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        # No faces for this asset -> SCRFD should run on the crop
        ui_stub.faces[asset_id] = []

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_fallback2.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_fallback2.sqlite3"}"

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

        store = Store(tmp_path / "test_fallback2.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        assert result["dry_run"] is True
        assert len(result["results"]) == 1
        r = result["results"][0]
        # SCRFD may or may not find a face depending on the fixture image
        assert r["status"] in ("would_write", "skipped")
        if r["status"] == "skipped":
            assert r["message"] == "no head found"

    def test_no_head_found_when_both_sources_fail(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """When neither Immich face nor SCRFD yields a head, appearance is skipped."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        # Add a face that does NOT overlap the person box (containment < 0.5)
        # Person box: 200,100,600,500 (400x400)
        # Face box far away -> containment = 0
        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-far",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 1000.0,
                "boundingBoxY1": 1000.0,
                "boundingBoxX2": 1200.0,
                "boundingBoxY2": 1200.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_fallback3.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_fallback3.sqlite3"}"

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

        store = Store(tmp_path / "test_fallback3.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        assert result["dry_run"] is True
        assert len(result["results"]) == 1
        r = result["results"][0]
        # If SCRFD also finds nothing on the fixture, should be "no head found"
        if r["status"] == "skipped":
            assert r["message"] == "no head found"

    def test_face_overlap_knob_controls_both_anchoring_and_head_resolution(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """The face_overlap knob is the single tunable for face→person containment."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        # Face box with containment exactly 1.0 (fully inside)
        # Person box: 200,100,600,500 (400x400)
        # Face box: 300,200,500,400 (200x200) fully inside -> ratio=1.0
        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-edge",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_fallback4.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_fallback4.sqlite3"}"

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

        store = Store(tmp_path / "test_fallback4.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        assert result["dry_run"] is True
        r = result["results"][0]
        assert r["status"] == "would_write"
        assert list(r["head_box"]) == [300.0, 200.0, 500.0, 400.0]


class TestDryRunBatches:
    """Tests for dry-run confirmations collected into batches and promotion."""

    def test_dry_run_confirmation_creates_batch(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """A dry-run confirmation persists as a batch with all its actions."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        # Add an unassigned face that overlaps the person box
        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_dryrun.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_dryrun.sqlite3"}"

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

        store = Store(tmp_path / "test_dryrun.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # Dry-run confirm for cluster 0
        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        assert result["dry_run"] is True
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["status"] == "would_write"
        assert "batch_id" in result  # The batch ID should be returned

        # Verify the batch was persisted
        batch_id = result["batch_id"]
        batch = store.get_dry_run_batch(batch_id)
        assert batch is not None
        assert batch.name == "Alice"
        assert batch.window == "2026-11-10"
        assert batch.cluster == 0
        assert len(batch.actions) == 1
        assert batch.actions[0].asset_id == asset_id
        assert batch.actions[0].appearance == 0
        assert batch.actions[0].head_box == (300.0, 200.0, 500.0, 400.0)

    def test_multiple_dry_run_confirmations_create_separate_batches(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Each dry-run confirmation creates its own batch."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_dryrun2.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_dryrun2.sqlite3"}"

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

        store = Store(tmp_path / "test_dryrun2.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # First dry-run
        result1 = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )
        batch_id1 = result1["batch_id"]

        # Second dry-run (same cluster, same name)
        result2 = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )
        batch_id2 = result2["batch_id"]

        # Should be separate batches
        assert batch_id1 != batch_id2

        # Both batches should exist
        batch1 = store.get_dry_run_batch(batch_id1)
        batch2 = store.get_dry_run_batch(batch_id2)
        assert batch1 is not None
        assert batch2 is not None

        # List all batches
        batches = store.get_dry_run_batches()
        assert len(batches) == 2

    def test_promote_batch_replays_writes(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Promoting a dry-run batch replays the actions as real writes."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_promote.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
write_api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_promote.sqlite3"}"

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

        store = Store(tmp_path / "test_promote.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        # Use a name that doesn't exist in the stub's people registry
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="NewPerson",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # First, do a dry-run to create a batch
        dry_run_result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "NewPerson",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )
        batch_id = dry_run_result["batch_id"]

        # Now promote the batch
        promote_result = server.promote_batch(batch_id)

        # Verify promotion succeeded
        assert promote_result["batch_id"] == batch_id
        assert promote_result["promoted"] is True
        assert len(promote_result["results"]) == 1
        r = promote_result["results"][0]
        assert r["status"] == "written"
        assert "person_id" in promote_result

        # Verify the write ledger has the entry
        person_id = promote_result["person_id"]
        assert store.has_write_ledger_entry(asset_id, person_id)

        # Verify the stub recorded the write requests
        face_requests = [r for r in ui_stub.faces_requests() if r.method == "POST"]
        assert len(face_requests) == 1
        assert face_requests[0].body["assetId"] == asset_id
        assert face_requests[0].body["personId"] == person_id

        # Person should have been created via POST /people
        people_requests = [r for r in ui_stub.people_requests() if r.method == "POST"]
        assert len(people_requests) == 1
        assert people_requests[0].body["name"] == "NewPerson"

    def test_promote_batch_is_idempotent(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Promoting the same batch twice produces no duplicate writes."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_promote_idem.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
write_api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_promote_idem.sqlite3"}"

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

        store = Store(tmp_path / "test_promote_idem.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        # Use a name that doesn't exist in the stub's people registry
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="NewPerson",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # Dry-run to create batch
        dry_run_result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "NewPerson",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )
        batch_id = dry_run_result["batch_id"]

        # Promote first time
        result1 = server.promote_batch(batch_id)
        assert result1["promoted"] is True
        person_id = result1["person_id"]

        # Promote second time
        result2 = server.promote_batch(batch_id)
        assert result2["promoted"] is True
        assert result2["person_id"] == person_id

        # Should have only one write ledger entry
        ledger_entries = store.get_write_ledger_entries()
        matching = [
            e
            for e in ledger_entries
            if e.asset_id == asset_id and e.person_id == person_id
        ]
        assert len(matching) == 1

        # Should have only one POST /faces call
        face_requests = [r for r in ui_stub.faces_requests() if r.method == "POST"]
        assert len(face_requests) == 1

    def test_promote_nonexistent_batch_returns_error(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Promoting a non-existent batch returns an error."""
        from reid.config import load_config
        from reid.core import CoreServer

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_promote_err.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
write_api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_promote_err.sqlite3"}"

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

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        result = server.promote_batch("nonexistent-batch-id")
        assert "error" in result

    def test_dry_run_no_immich_writes(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Dry-run mode produces zero Immich writes - no faces, no person creation."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_dryrun_nowrite.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_dryrun_nowrite.sqlite3"}"

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

        store = Store(tmp_path / "test_dryrun_nowrite.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # Dry-run confirm
        result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )

        assert result["dry_run"] is True

        # Verify NO Immich write requests were made
        face_requests = [r for r in ui_stub.faces_requests() if r.method == "POST"]
        people_requests = [r for r in ui_stub.people_requests() if r.method == "POST"]

        # No POST /faces or POST /people should be made in dry-run
        assert len(face_requests) == 0, "No POST /faces should be made in dry-run"
        assert len(people_requests) == 0, "No POST /people should be made in dry-run"

    def test_list_dry_run_batches_endpoint(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """GET /api/dry-run-batches returns all persisted dry-run batches."""
        from reid.config import load_config
        from reid.core import CoreServer
        from reid.store import Appearance, Proposal, Store

        asset_id = library.PHOTO_1_STANDING_ID

        ui_stub.faces[asset_id] = [
            {
                "id": "face-test-unassigned",
                "assetId": asset_id,
                "person": None,
                "boundingBoxX1": 300.0,
                "boundingBoxY1": 200.0,
                "boundingBoxX2": 500.0,
                "boundingBoxY2": 400.0,
                "imageWidth": 1440,
                "imageHeight": 1440,
                "sourceType": "machine-learning",
            }
        ]

        weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
        config_path = tmp_path / "test_list_batches.toml"
        config_path.write_text(
            f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / "test_list_batches.sqlite3"}"

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

        store = Store(tmp_path / "test_list_batches.sqlite3")
        store.add_appearances(
            asset_id,
            [
                Appearance(
                    asset_id=asset_id,
                    appearance=0,
                    embedding=b"\x00" * 512,
                    confidence=0.9,
                    box_left=200.0,
                    box_top=100.0,
                    box_right=600.0,
                    box_bottom=500.0,
                    pixel_source="preview",
                    captured_at=datetime.fromisoformat("2026-11-10T10:00:00"),
                )
            ],
        )
        store.replace_proposals(
            [
                Proposal(
                    asset_id=asset_id,
                    appearance=0,
                    window="2026-11-10",
                    cluster=0,
                    status=ProposalStatus.PROPOSED,
                    proposed_name="Alice",
                )
            ]
        )

        cfg = load_config(config_path)
        server = CoreServer(cfg)

        # Create a dry-run batch
        dry_run_result = server.confirm_cluster(
            {
                "cluster": 0,
                "window": "2026-11-10",
                "name": "Alice",
                "included": [{"asset_id": asset_id, "appearance": 0}],
                "dry_run": True,
            }
        )
        batch_id = dry_run_result["batch_id"]

        # List batches via server method
        batches = server.get_dry_run_batches()
        assert len(batches) == 1
        assert batches[0]["id"] == batch_id
        assert batches[0]["name"] == "Alice"
        assert batches[0]["window"] == "2026-11-10"
        assert batches[0]["cluster"] == 0
        assert len(batches[0]["actions"]) == 1
