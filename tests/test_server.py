"""Tests for the confirmation UI HTTP API server."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from reid.config import load_config
from reid.server import UIServer, serve
from reid.store import Store, Proposal, Appearance, NEW_ASSET_FEED
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
    config_path = tmp_path / "uitest.toml"
    config_path.write_text(f"""
[immich]
base_url = "{ui_stub.base_url}"
api_key = "{API_KEY}"
page_size = 1000

[store]
path = "{tmp_path / 'uitest.sqlite3'}"

[detection]
pixel_source = "preview"
confidence_floor = 0.35
min_box_height = 100

[clustering]
window_days = 1
distance_threshold = 0.5
min_cluster_size = 2
face_overlap = 0.5

[run]
log_level = "info"
""")
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
            captured_at="2026-11-10T10:00:00",
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
            captured_at="2026-11-10T10:00:00",
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
            captured_at="2026-11-10T11:00:00",
        ),
    ]
    store.add_appearances("asset1", appearances[:2])
    store.add_appearances("asset2", appearances[2:])

    # Add proposals (window "2026-11-10", cluster 0 and 1)
    proposals = [
        Proposal(asset_id="asset1", appearance=0, window="2026-11-10", cluster=0, status="proposed", proposed_name="Alice"),
        Proposal(asset_id="asset1", appearance=1, window="2026-11-10", cluster=0, status="proposed", proposed_name="Alice"),
        Proposal(asset_id="asset2", appearance=0, window="2026-11-10", cluster=1, status="ask_name", proposed_name=None),
    ]
    store.replace_proposals(proposals)

    return store


class TestServerAPI:
    """Test the UI server API methods directly."""

    def test_get_ui_state(self, config_with_store: Path, populated_store: Store) -> None:
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

    def test_get_evidence(self, config_with_store: Path, populated_store: Store) -> None:
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

    def test_get_config_knobs(self, config_with_store: Path, populated_store: Store) -> None:
        cfg = load_config(config_with_store)
        server = UIServer(cfg)

        knobs = server.get_config_knobs()

        assert knobs.window_days == 1
        assert knobs.distance_threshold == 0.5
        assert knobs.confidence_floor == 0.35
        assert knobs.min_box_height == 100.0
        assert knobs.face_overlap == 0.5

    def test_get_proposals(self, config_with_store: Path, populated_store: Store) -> None:
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

        server_thread = threading.Thread(target=serve, args=(config_with_store, "127.0.0.1", port), daemon=True)
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