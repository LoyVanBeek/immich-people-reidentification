"""Tests for the confirmation UI HTTP API server."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

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
        assert state["core_commit"]  # git commit indicator

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

    def test_ui_with_core_only_config(
        self, tmp_path: Path, populated_store: Store
    ) -> None:
        """The UI can run from a [core]-only config (no [immich]/[store]):
        this is what an independent UI on another machine uses."""
        import socket
        import threading

        from reid.server import serve_ui

        def free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

        core_port = free_port()
        ui_port = free_port()

        # [core]-only config: enough for the UI to find and auth to the core.
        core_config = tmp_path / "core.toml"
        core_config.write_text(
            f"""
[core]
url = "http://127.0.0.1:{core_port}"
api_key = "core-key"
host = "127.0.0.1"
port = {core_port}
"""
        )

        # The core needs the full config (with [immich]/[store]) to own the
        # store, but it must bind the same port the UI points at.
        full_config = tmp_path / "full.toml"
        full_config.write_text(
            f"""
[immich]
base_url = "http://immich:2283"
api_key = "k"

[store]
path = "{tmp_path / "store.sqlite3"}"

[head_detection]
weights_path = "{Path(__file__).parent.parent / "src" / "reid" / "weights" / "scrfd_500m.onnx"}"

[core]
url = "http://127.0.0.1:{core_port}"
api_key = "core-key"
host = "127.0.0.1"
port = {core_port}
"""
        )

        load_config(full_config)

        threading.Thread(
            target=serve_core,
            args=(full_config, "127.0.0.1", core_port),
            daemon=True,
        ).start()

        threading.Thread(
            target=serve_ui,
            args=(core_config, "127.0.0.1", ui_port),
            daemon=True,
        ).start()

        for _ in range(50):
            try:
                with urlopen(
                    f"http://127.0.0.1:{ui_port}/api/state", timeout=1
                ) as resp:
                    assert resp.status == 200
                    data = json.load(resp)
                    assert "clusters" in data
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            pytest.fail("UI did not start")

        # index.html carries the version stamp and the injected config with
        # the UI's git commit (immich_url not set here, so it is null).
        with urlopen(f"http://127.0.0.1:{ui_port}/", timeout=10) as resp:
            assert resp.status == 200
            html = resp.read().decode()
            assert "__UI_VERSION__" not in html
            assert '"commit":' in html
            assert "window.__REID_UI__" in html

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

    def test_http_jobs_endpoint_reports_idle(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/jobs") as resp:
            assert resp.status == 200
            data = json.load(resp)
        assert data["active"] is False
        assert data["last_run"] is None

    def test_http_start_job_endpoint(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reid.core import JobManager

        started: list[str] = []

        def fake_start(self, trigger: str) -> dict[str, object]:
            started.append(trigger)
            return {"job_id": "run-1", "pid": 42}

        monkeypatch.setattr(JobManager, "start", fake_start)
        request = Request(
            f"{server_url}/api/jobs",
            data=json.dumps({"trigger": "manual"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as resp:
            assert resp.status == 200
            data = json.load(resp)

        assert data == {"job_id": "run-1", "pid": 42}
        assert started == ["manual"]

    def test_http_state_endpoint(self, server_url: str) -> None:
        with urlopen(f"{server_url}/api/state") as resp:
            assert resp.status == 200
            data = json.load(resp)
            assert "clusters" in data
            assert len(data["clusters"]) == 2
            assert data["core_commit"]  # git commit indicator for the UI chip

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
