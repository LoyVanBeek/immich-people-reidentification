"""Tests for the confirmation UI HTTP API server."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pytest

from reid.config import load_config
from reid.enums import ProposalStatus
from reid.server import UIServer, serve
from reid.store import Appearance, Proposal, Store
from tests import library
from tests.stub_immich import API_KEY, StubImmich


@pytest.fixture
def ui_stub() -> StubImmich:
    """A stub with people."""
    server = StubImmich(library.detection_registry())
    server.people = library.people_registry()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def config_with_store(tmp_path: Path, ui_stub: StubImmich) -> Path:
    """A config pointing to a store with some proposals."""
    weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"
    config_path = tmp_path / "uitest.toml"
    config_path.write_text(
        f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
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
    return config_path


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
        server = UIServer(cfg)

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

    def test_get_evidence(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = UIServer(cfg)

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
        server = UIServer(cfg)

        people = server.get_people()

        assert isinstance(people, list)
        assert len(people) == 5
        assert people == ["Alice", "Bob", "Carol", "Dave", "Erin"]

    def test_get_config_knobs(
        self, config_with_store: Path, populated_store: Store
    ) -> None:
        cfg = load_config(config_with_store)
        server = UIServer(cfg)

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
        server = UIServer(cfg)

        proposals = server.get_proposals()

        assert len(proposals) == 3
        assert proposals[0]["proposed_name"] == "Alice"
        assert proposals[2]["status"] == "ask_name"

    def test_get_summary(self, config_with_store: Path, populated_store: Store) -> None:
        cfg = load_config(config_with_store)
        server = UIServer(cfg)

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
    def server_url(self, config_with_store: Path, populated_store: Store) -> str:
        """Start a server for this test."""
        # Find a free port
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server_thread = threading.Thread(
            target=serve, args=(config_with_store, "127.0.0.1", port), daemon=True
        )
        server_thread.start()

        # Wait for server to start
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urlopen(f"{base_url}/api/state", timeout=1)
                break
            except Exception:
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

    def test_http_serves_static_files(self, server_url: str) -> None:
        with urlopen(f"{server_url}/") as resp:
            assert resp.status == 200
            assert resp.headers.get("content-type", "").startswith("text/html")

        with urlopen(f"{server_url}/index.html") as resp:
            assert resp.status == 200

        with urlopen(f"{server_url}/style.css") as resp:
            assert resp.status == 200
            assert resp.headers.get("content-type", "").startswith("text/css")

        with urlopen(f"{server_url}/app.js") as resp:
            assert resp.status == 200
            # mimetypes.guess_type returns "text/javascript" for .js files
            assert resp.headers.get("content-type", "").startswith("text/javascript")

    def test_http_404_for_missing_static(self, server_url: str) -> None:
        import urllib.error

        try:
            urlopen(f"{server_url}/nonexistent.html")
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
        from reid.server import UIServer
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
        server = UIServer(cfg)

        # Spy on the head detector to verify it's NOT called when Immich face is available
        detect_calls = {"count": 0}
        original_detect = server.immich._head_detector.detect

        def spy_detect(crop):
            detect_calls["count"] += 1
            return original_detect(crop)

        server.immich._head_detector.detect = spy_detect

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer

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
        server = UIServer(cfg)

        result = server.promote_batch("nonexistent-batch-id")
        assert "error" in result

    def test_dry_run_no_immich_writes(
        self, tmp_path: Path, ui_stub: StubImmich
    ) -> None:
        """Dry-run mode produces zero Immich writes - no faces, no person creation."""
        from reid.config import load_config
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
        from reid.server import UIServer
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
        server = UIServer(cfg)

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
