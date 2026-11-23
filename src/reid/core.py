"""Core API server: the single owner of the store.

Every store-reading and store-writing endpoint lives here, behind an
optional API-key check. The UI server (reid-ui) serves the static
assets and proxies /api/* to this process, so the browser never needs
direct access to the store and the UI container needs no data volume.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .anchoring import claim_face, face_person_overlap
from .config import Config, load_config
from .enums import (
    ClusterRowStatus,
    DryRunActionStatus,
    EvidenceCardStatus,
    ProposalStatus,
    WriteResultStatus,
)
from .head_detector import HeadDetectorConfig
from .immich import Face, ImmichClient, ImmichError, isoformat_z
from .store import (
    Appearance,
    DryRunAction,
    Proposal,
    Store,
    WriteLedgerEntry,
)

_log = logging.getLogger("reid.core")

# Crop outline colours, a colourblind-safe blue/orange pair, in BGR as cv2
# expects: the target's box in blue, every other appearance's box in orange.
TARGET_OUTLINE_COLOR: tuple[int, int, int] = (255, 140, 79)
NEIGHBOUR_OUTLINE_COLOR: tuple[int, int, int] = (0, 165, 255)


@dataclass(frozen=True)
class ClusterRow:
    """One row in the review queue sidebar."""

    cluster: int
    window: str
    proposed_name: str | None
    anchor: str | None
    confidence: float
    appearances: int
    writable: int
    conflicted: int
    ask_for_name: int
    labeled: int
    status: ClusterRowStatus
    crops: list[dict[str, Any]]  # appearance refs; UI fetches each via /api/crop
    fully_labeled: bool = False  # every appearance already labeled by Immich


@dataclass(frozen=True)
class EvidenceCard:
    """One appearance card in the evidence grid."""

    asset_id: str
    appearance: int
    window: str
    cluster: int
    date: str  # ISO format date string for JSON serialization
    confidence: float
    immich_label: str | None
    status: EvidenceCardStatus
    included: bool  # checkbox state for writable appearances
    pixel_source: str  # "preview" | "original"
    pin: str | None = None  # operator's pinned person, if any
    conflict_names: list[str] | None = None  # labels fighting in a conflicted cluster
    asset_name: str = ""  # original file name, for human-readable review
    ignored: bool = False  # operator flagged this detection as wrong
    proposed_name: str | None = None  # name this proposal would write


@dataclass(frozen=True)
class AssignmentPane:
    """Right pane: name prompt, stats, confirm/skip."""

    cluster: int
    window: str
    proposed_name: str | None
    anchor: str | None
    total_appearances: int
    writable: int
    conflicted: int
    ask_for_name: int
    labeled: int
    existing_people: list[str]


@dataclass(frozen=True)
class StatsPane:
    """Per-window stats from the latest run."""

    window: str
    appearances: int
    clusters: int
    anchored_pct: float
    conflicted_pct: float
    anchor_less_pct: float
    singletons: int
    proposals: int
    ask_for_name: int


@dataclass(frozen=True)
class UIConfigKnobs:
    """The five tuning knobs exposed in the ⚙ dropdown."""

    window_days: int
    distance_threshold: float
    confidence_floor: float
    min_box_height: float
    face_overlap: float


@dataclass(frozen=True)
class UIState:
    """Complete UI state for one request."""

    clusters: list[ClusterRow]
    active_cluster: int | None
    evidence: list[EvidenceCard]
    assignment: AssignmentPane | None
    stats: list[StatsPane]
    config_knobs: UIConfigKnobs
    people: list[str]  # existing Immich people for autocomplete
    dry_run: bool
    immich_url: str  # base URL for "open in Immich" viewer links


class CoreAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the core API (all store-owning endpoints)."""

    # Class variable set by serve_core()
    core_server: CoreServer | None = None

    def do_GET(self) -> None:
        if not self._authorized():
            return
        try:
            self._do_get()
        except Exception as exc:  # noqa: BLE001
            # Never let a handler exception close the connection silently:
            # the browser would show a blank page / network error with no clue.
            _log.exception("GET %s failed", self.path)
            self._send_json({"error": f"internal server error: {exc}"}, status=500)

    def _do_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._send_json(self.core_server.get_ui_state())  # type: ignore[union-attr]
        elif path == "/api/jobs":
            server = self.core_server
            assert server is not None
            self._send_json(
                {
                    "active": server.jobs.active(),
                    "last_run": server.store.latest_run(),
                }
            )
        elif path == "/api/write-review":
            self._send_json(self.core_server.get_write_review())  # type: ignore[union-attr]
        elif path == "/api/evidence":
            self._send_json(self.core_server.get_evidence(parse_qs(parsed.query)))  # type: ignore[union-attr]
        elif path == "/api/people":
            self._send_json({"people": self.core_server.get_people()})  # type: ignore[union-attr]
        elif path == "/api/config":
            self._send_json(asdict(self.core_server.get_config_knobs()))  # type: ignore[union-attr]
        elif path == "/api/proposals":
            self._send_json(self.core_server.get_proposals())  # type: ignore[union-attr]
        elif path == "/api/summary":
            self._send_json(self.core_server.get_summary())  # type: ignore[union-attr]
        elif path == "/api/dry-run-batches":
            self._send_json(self.core_server.get_dry_run_batches())  # type: ignore[union-attr]
        elif path == "/api/crop":
            try:
                data, ctype = self.core_server.get_crop(parse_qs(parsed.query))  # type: ignore[union-attr]
            except (ValueError, LookupError) as exc:
                self.send_error(404, str(exc))
                return
            self._send_bytes(data, ctype)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if not self._authorized():
            return
        try:
            self._do_post()
        except Exception as exc:  # noqa: BLE001
            _log.exception("POST %s failed", self.path)
            self._send_json({"error": f"internal server error: {exc}"}, status=500)

    def _do_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/confirm":
            content_length = int(self.headers.get("content-length", "0"))
            body_bytes = self.rfile.read(content_length)
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            result = self.core_server.confirm_cluster(body)  # type: ignore[union-attr]
            self._send_json(result)
        elif path == "/api/skip":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.dismiss_cluster(  # type: ignore[union-attr]
                str(body.get("window", "")),
                int(body.get("cluster", -1)),
                bool(body.get("dismissed", True)),
            )
            self._send_json(result)
        elif path == "/api/reject":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.reject_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                str(body.get("name", "")).strip(),
            )
            self._send_json(result)
        elif path == "/api/pin":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.pin_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                str(body.get("name", "")).strip(),
            )
            self._send_json(result)
        elif path == "/api/unpin":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.unpin_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
            )
            self._send_json(result)
        elif path == "/api/unreject":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.unreject_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                str(body.get("name", "")).strip(),
            )
            self._send_json(result)
        elif path == "/api/jobs":
            # A bare POST (no body) triggers with the default label.
            body = (
                self._read_json()
                if self.headers.get("content-length", "0") != "0"
                else {}
            )
            if body is None:
                return
            server = self.core_server
            assert server is not None
            result = server.jobs.start(str(body.get("trigger", "manual")))
            self._send_json(result)
        elif path == "/api/jobs/cancel":
            server = self.core_server
            assert server is not None
            self._send_json(server.jobs.cancel())
        elif path == "/api/write-decision":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.set_write_decision(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                bool(body.get("included", True)),
            )
            self._send_json(result)
        elif path == "/api/confirm-all":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.confirm_all(body)  # type: ignore[union-attr]
            self._send_json(result)
        elif path == "/api/promote-all":
            self._send_json(self.core_server.promote_all())  # type: ignore[union-attr]
        elif path == "/api/write-pending":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.write_pending(body)  # type: ignore[union-attr]
            self._send_json(result)
        elif path == "/api/ignore":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.ignore_appearance_request(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
            )
            self._send_json(result)
        elif path == "/api/unignore":
            body = self._read_json()
            if body is None:
                return
            result = self.core_server.unignore_appearance_request(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
            )
            self._send_json(result)
        elif path == "/api/config":
            body = self._read_json()
            if body is None:
                return
            self._send_json(self.core_server.save_config_knobs(body))  # type: ignore[union-attr]
        elif path.startswith("/api/promote/"):
            batch_id = path[len("/api/promote/") :]
            if not batch_id:
                self.send_error(400, "Missing batch ID")
                return
            result = self.core_server.promote_batch(batch_id)  # type: ignore[union-attr]
            self._send_json(result)
        else:
            self.send_error(405, "Method not allowed")

    def _read_json(self) -> dict[str, Any] | None:
        """Read and parse the JSON request body; answers 400 on failure."""
        content_length = int(self.headers.get("content-length", "0"))
        body_bytes = self.rfile.read(content_length)
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return None
        if not isinstance(body, dict):
            self.send_error(400, "JSON body must be an object")
            return None
        return body

    def log_message(self, fmt: str, *args: Any) -> None:
        _log.debug(fmt % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        """Requests must carry the core token when one is configured."""
        server = self.core_server
        if server is None:
            return False
        key = server.cfg.core.api_key
        return not key or self.headers.get("x-api-key") == key


class JobManager:
    """Runs the pipeline as an isolated subprocess, one job at a time.

    The subprocess is the same `reid-batch` CLI the nightly timer used to
    run directly — one code path, isolated so a pipeline crash can never
    take the core (or the UI) down. Progress lands in the run_status
    table via the REID_RUN_ID env var.
    """

    def __init__(self, store: Store, config_path: Path) -> None:
        self._store = store
        self._config_path = config_path
        # No subprocess survives a core restart: close stale run rows so the
        # UI never shows a run that is not actually happening.
        store.mark_interrupted_runs()
        self._proc: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        # The watcher thread cannot touch SQLite (the connection belongs to
        # the thread that created it), so it only records the exit code; the
        # next call from the server thread finalizes the run row.
        self._pending_exit: tuple[str, int] | None = None

    def active(self) -> bool:
        """True while a pipeline subprocess is running."""
        self._finalize()
        return self._proc is not None

    def _finalize(self) -> None:
        """Write the watcher's exit code to the store, on the server thread."""
        pending = self._pending_exit
        if pending is None:
            return
        self._pending_exit = None
        run_id, code = pending
        if run_id != self._job_id:
            return  # cancelled or superseded: cancel() already recorded it
        state = "done" if code == 0 else "failed"
        error = None if code == 0 else f"pipeline exited with code {code}"
        self._store.finish_run(run_id, state, datetime.now(UTC), error)
        self._proc = None
        self._job_id = None

    def start(self, trigger: str) -> dict[str, Any]:
        """Start one pipeline run; refuses while another is active."""
        if self.active():
            return {"error": "a pipeline run is already in progress"}
        run_id = str(uuid.uuid4())
        env = {**os.environ, "REID_RUN_ID": run_id}
        proc = subprocess.Popen(
            # `python -m reid.cli` runs the batch entry point itself (the
            # module has no subcommands); using -m avoids PATH assumptions.
            [
                sys.executable,
                "-m",
                "reid.cli",
                "--config",
                str(self._config_path),
            ],
            env=env,
            # Inherit the core's stdout/stderr so `docker logs reid-core`
            # shows the pipeline's own log lines alongside the core's.
        )
        self._proc = proc
        self._job_id = run_id
        self._store.start_run(run_id, trigger, proc.pid, datetime.now(UTC))
        threading.Thread(target=self._watch, args=(proc, run_id), daemon=True).start()
        return {"job_id": run_id, "pid": proc.pid}

    def _watch(self, proc: subprocess.Popen[bytes], run_id: str) -> None:
        """Record the subprocess exit; the server thread writes it to the store.

        No SQLite here: the connection is thread-bound to the server thread.
        """
        code = proc.wait()
        self._pending_exit = (run_id, code)

    def cancel(self) -> dict[str, Any]:
        """Terminate the running job; checkpointing makes it resumable."""
        if self._proc is None or self._job_id is None:
            return {"error": "no pipeline run in progress"}
        proc, run_id = self._proc, self._job_id
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # Drop any exit the watcher recorded: this run is cancelled, and the
        # row must not flip to failed afterwards.
        self._pending_exit = None
        self._store.finish_run(run_id, "cancelled", datetime.now(UTC))
        self._proc = None
        self._job_id = None
        return {"cancelled": run_id}


class CoreServer:
    """The core API server: the single owner of the store."""

    def __init__(self, cfg: Config, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.store = Store(cfg.store_path)
        # Original file names, fetched once per asset and cached for the
        # process lifetime: the review UI shows file names, not UUIDs.
        self._asset_names: dict[str, str] = {}
        # The pipeline runs as an isolated subprocess, one at a time.
        self.jobs = JobManager(self.store, config_path or Path("reid.toml"))
        head_cfg = HeadDetectorConfig(
            weights_path=cfg.head_detection.weights_path,
            input_size=cfg.head_detection.input_size,
            confidence_threshold=cfg.head_detection.confidence_threshold,
            top_expansion_ratio=cfg.head_detection.top_expansion_ratio,
            width_expansion_ratio=cfg.head_detection.width_expansion_ratio,
            nms_threshold=cfg.head_detection.nms_threshold,
        )
        self.immich = ImmichClient(
            cfg.immich.base_url,
            cfg.immich.api_key,
            page_size=cfg.immich.page_size,
            timeout=cfg.immich.timeout_seconds,
            head_detector_cfg=head_cfg,
        )
        self._dry_run = True  # default to dry-run; toggled via UI

    def get_ui_state(self) -> dict[str, Any]:
        """Assemble the complete UI state for the initial page load."""
        proposals = self.store.proposals()
        appearances_by_asset = self.store.appearances_by_asset()

        # Group proposals by window + cluster
        clusters = self._build_cluster_rows(proposals, appearances_by_asset)
        # Cluster numbers repeat across windows, so the default selection is
        # the first ROW — window included — and the panes are built for that
        # exact window only.
        active = clusters[0] if clusters else None
        evidence = (
            self._build_evidence_for_cluster(
                active.cluster, proposals, appearances_by_asset, window=active.window
            )
            if active is not None
            else []
        )
        assignment = (
            self._build_assignment(
                active.cluster, proposals, appearances_by_asset, window=active.window
            )
            if active is not None
            else None
        )
        stats = self._build_stats()
        config_knobs = self.get_config_knobs()
        people = self.get_people()

        state = UIState(
            clusters=clusters,
            active_cluster=active.cluster if active is not None else None,
            evidence=evidence,
            assignment=assignment,
            stats=stats,
            config_knobs=config_knobs,
            people=people,
            dry_run=self._dry_run,
            immich_url=self.cfg.immich.base_url.rstrip("/"),
        )
        return asdict(state)

    def get_crop(self, query: dict[str, list[str]]) -> tuple[bytes, str]:
        """Serve one appearance's person crop as JPEG, cropped from Immich pixels.

        Pixels come from the configured preview/original source — the same
        source detection ran on, so appearance boxes match without any
        coordinate mapping.

        Optional params:
        - ``expand``: fraction of the box size to grow on each side, for
          context around the person (e.g. 0.6 = 60% margin).
        - ``outline``: draw the person's box as a border inside the crop, so
          the reviewer can tell the target apart from neighbours.
        - ``neighbours``: draw every other appearance's box on the same
          asset, in a contrasting colour, wherever that box reaches into
          the card. Off by default so mini crops stay clean.
        """
        import cv2

        asset_id = (query.get("asset_id") or [""])[0]
        appearance_raw = (query.get("appearance") or [""])[0]
        if not asset_id or not appearance_raw:
            raise ValueError("asset_id and appearance are required")
        appearance_no = int(appearance_raw)
        expand = float((query.get("expand") or ["0"])[0] or 0)
        outline = (query.get("outline") or ["0"])[0].lower() not in ("0", "", "false")
        neighbours = (query.get("neighbours") or ["0"])[0].lower() not in (
            "0",
            "",
            "false",
        )

        app_list = self.store.appearances_by_asset().get(asset_id, [])
        appearance = next((a for a in app_list if a.appearance == appearance_no), None)
        if appearance is None:
            raise LookupError(f"appearance not found: {asset_id}#{appearance_no}")

        pixels = self.immich.fetch_pixels(
            asset_id, source=self.cfg.detection.pixel_source
        )
        image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("pixels are not a decodable image")

        left, top, right, bottom = (int(v) for v in appearance.box)
        height, width = image.shape[:2]
        if expand > 0:
            grow_x = int((right - left) * expand)
            grow_y = int((bottom - top) * expand)
            x0, y0 = max(left - grow_x, 0), max(top - grow_y, 0)
            x1, y1 = min(right + grow_x, width), min(bottom + grow_y, height)
        else:
            x0, y0 = max(left, 0), max(top, 0)
            x1, y1 = min(right, width), min(bottom, height)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError("appearance box is outside the image")

        thickness = max(2, min(crop.shape[0], crop.shape[1]) // 80)
        if neighbours:
            # Every other appearance's box that reaches into the card, in a
            # contrasting colour. The reviewer's question is "which person do
            # these pixels belong to", and any neighbour visible in the card
            # can be mistaken for the target — not just neighbours whose box
            # intersects the target's box. Drawn before the target's own
            # outline, which stays on top where the boxes overlap.
            for other in app_list:
                if other.appearance == appearance_no:
                    continue
                nl, nt, nr, nb = other.box
                if nl < x1 and nr > x0 and nt < y1 and nb > y0:
                    cv2.rectangle(
                        crop,
                        (int(nl - x0), int(nt - y0)),
                        (int(nr - x0), int(nb - y0)),
                        NEIGHBOUR_OUTLINE_COLOR,
                        thickness,
                    )
        if outline:
            # The person's box, translated into crop coordinates.
            bx0, by0 = max(left - x0, 0), max(top - y0, 0)
            bx1 = min(right - x0, x1 - x0 - 1)
            by1 = min(bottom - y0, y1 - y0 - 1)
            cv2.rectangle(crop, (bx0, by0), (bx1, by1), TARGET_OUTLINE_COLOR, thickness)

        max_side = 320
        scale = min(1.0, max_side / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("could not encode the crop as JPEG")
        return buf.tobytes(), "image/jpeg"

    def get_evidence(self, query: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Return evidence cards for a specific cluster."""
        cluster_q = query.get("cluster", [""])[0]
        window_q = query.get("window", [""])[0]
        if not cluster_q or not window_q:
            return []
        cluster = int(cluster_q)
        proposals = self.store.proposals()
        appearances_by_asset = self.store.appearances_by_asset()
        # Immich's own face labels for these assets: what the appearance is
        # currently called in Immich, if anything.
        faces_by_asset: dict[str, list[Face]] = {}
        for asset_id in {p.asset_id for p in proposals}:
            try:
                faces_by_asset[asset_id] = self.immich.fetch_faces(asset_id)
            except Exception as exc:  # noqa: BLE001
                _log.warning("failed to fetch faces for %s: %s", asset_id, exc)
                faces_by_asset[asset_id] = []
        evidence = self._build_evidence_for_cluster(
            cluster,
            proposals,
            appearances_by_asset,
            window_q,
            faces_by_asset=faces_by_asset,
            rejected=set(self.store.appearance_rejections()),
        )
        return [asdict(e) for e in evidence]

    def get_people(self) -> list[str]:
        """Return sorted list of existing Immich person names for autocomplete.

        Failures (Immich unreachable, bad key, parsing) degrade to an empty
        list: the UI must still render the queue and evidence without names.
        """
        try:
            people = self.immich.fetch_people()
            return sorted(p.name for p in people)
        except Exception as exc:  # noqa: BLE001
            _log.warning("failed to fetch people: %s", exc)
            return []

    def get_config_knobs(self) -> UIConfigKnobs:
        """Return the five tuning knobs: TOML values overridden by the store."""
        overrides = self.store.tuning_values()
        return UIConfigKnobs(
            window_days=self._knob(
                overrides, "window_days", self.cfg.clustering.window_days, int
            ),
            distance_threshold=self._knob(
                overrides,
                "distance_threshold",
                self.cfg.clustering.distance_threshold,
                float,
            ),
            confidence_floor=self._knob(
                overrides,
                "confidence_floor",
                self.cfg.detection.confidence_floor,
                float,
            ),
            min_box_height=self._knob(
                overrides, "min_box_height", self.cfg.detection.min_box_height, float
            ),
            face_overlap=self._knob(
                overrides, "face_overlap", self.cfg.clustering.face_overlap, float
            ),
        )

    @staticmethod
    def _knob(overrides: dict[str, str], key: str, default: Any, cast: type) -> Any:
        """One tuning value: store override wins, falls back to config."""
        raw = overrides.get(key)
        if raw is None:
            return default
        try:
            return cast(raw)
        except ValueError:
            _log.warning("ignoring malformed tuning override %s=%r", key, raw)
            return default

    def save_config_knobs(self, body: dict[str, Any]) -> dict[str, Any]:
        """Persist operator tuning overrides; they apply to the next batch run."""
        allowed = {
            "window_days": (1, 30, int),
            "distance_threshold": (0.05, 1.0, float),
            "confidence_floor": (0.0, 1.0, float),
            "min_box_height": (20, 500, float),
            "face_overlap": (0.1, 1.0, float),
        }
        saved = []
        for key, (low, high, cast) in allowed.items():
            if key not in body:
                continue
            try:
                value = cast(body[key])
            except (TypeError, ValueError):
                return {"error": f"{key} must be a number"}
            if not low <= value <= high:
                return {"error": f"{key} must be between {low} and {high}"}
            self.store.set_tuning_value(key, repr(value))
            saved.append(key)
        if not saved:
            return {"error": "no tuning keys in request"}
        return {"saved": saved, "knobs": asdict(self.get_config_knobs())}

    def dismiss_cluster(
        self, window: str, cluster: int, dismissed: bool
    ) -> dict[str, Any]:
        """Persist 'I don't care about this cluster'."""
        self.store.dismiss_cluster(window, cluster, dismissed)
        return {"window": window, "cluster": cluster, "dismissed": dismissed}

    def reject_appearance(
        self, asset_id: str, appearance: int, proposed_name: str
    ) -> dict[str, Any]:
        """Record 'this appearance is not this person' and update the proposal."""
        if not proposed_name:
            return {"error": "a proposed name is required to reject an appearance"}
        self.store.add_appearance_rejection(asset_id, appearance, proposed_name)
        self.store.update_proposal_status(asset_id, appearance, ProposalStatus.REJECTED)
        return {"asset_id": asset_id, "appearance": appearance, "status": "rejected"}

    def pin_appearance(
        self, asset_id: str, appearance: int, name: str
    ) -> dict[str, Any]:
        """Record 'this appearance is this person' as an operator decision.

        The pin survives batch runs: anchoring proposes the pinned name for
        the appearance through the normal review path — nothing writes to
        Immich here.
        """
        name = name.strip()
        if not name:
            return {"error": "a person name is required to assign an appearance"}
        self.store.set_pin(asset_id, appearance, name)
        return {"asset_id": asset_id, "appearance": appearance, "pin": name}

    def unpin_appearance(self, asset_id: str, appearance: int) -> dict[str, Any]:
        """Remove the operator's pin; normal anchoring applies again."""
        self.store.delete_pin(asset_id, appearance)
        return {"asset_id": asset_id, "appearance": appearance, "pin": None}

    def ignore_appearance_request(
        self, asset_id: str, appearance: int
    ) -> dict[str, Any]:
        """Record 'this detection is wrong — never label it'.

        The ignore survives batch runs: the appearance is excluded from
        clustering entirely. Nothing writes to Immich here.
        """
        self.store.ignore_appearance(asset_id, appearance)
        return {"asset_id": asset_id, "appearance": appearance, "ignored": True}

    def unignore_appearance_request(
        self, asset_id: str, appearance: int
    ) -> dict[str, Any]:
        """Remove the ignore; normal clustering applies again."""
        self.store.unignore_appearance(asset_id, appearance)
        return {"asset_id": asset_id, "appearance": appearance, "ignored": False}

    def get_write_review(self) -> dict[str, Any]:
        """Person-centric review of everything about to be written.

        One tab per name with pending writes (across all days and clusters),
        that person's already-labeled appearances alongside (matched via
        face claims), and an "unnamed" tab for clusters still needing a
        name. Cards carry the effective write decision (persisted choice,
        defaulting to the proposal's writability). The pending section
        lists approved-but-unwritten items from dry-run batches for the
        per-item write flow.
        """
        proposals = self.store.proposals()
        # Approved-but-unwritten items: un-promoted batches, un-written actions.
        pending: list[dict[str, Any]] = []
        for batch in self.store.get_dry_run_batches():
            if batch.promoted_at is not None:
                continue
            for action in batch.actions:
                if action.written_at is not None:
                    continue
                pending.append(
                    {
                        "batch_id": batch.id,
                        "name": batch.name,
                        "window": batch.window,
                        "cluster": batch.cluster,
                        "asset_id": action.asset_id,
                        "appearance": action.appearance,
                        "asset_name": self._asset_name(action.asset_id),
                        "status": action.status.value,
                        "default_included": (
                            action.status == DryRunActionStatus.WOULD_WRITE
                        ),
                    }
                )
        if not proposals:
            return {
                "tabs": [],
                "unnamed": [],
                "total_writable": 0,
                "pending": pending,
                "total_pending": len(pending),
            }
        appearances_by_asset = self.store.appearances_by_asset()
        decisions = self.store.write_decisions()
        rejected = set(self.store.appearance_rejections())

        faces_by_asset: dict[str, list[Face]] = {}
        for asset_id in sorted({p.asset_id for p in proposals}):
            try:
                faces_by_asset[asset_id] = self.immich.fetch_faces(asset_id)
            except Exception as exc:  # noqa: BLE001
                _log.warning("failed to fetch faces for %s: %s", asset_id, exc)
                faces_by_asset[asset_id] = []

        # Build cards with the per-cluster logic, then regroup by person.
        all_cards: list[dict[str, Any]] = []
        groups: dict[tuple[str, int], list[Proposal]] = {}
        for p in proposals:
            groups.setdefault((p.window, p.cluster), []).append(p)
        for window, cluster in sorted(groups):
            cards = self._build_evidence_for_cluster(
                cluster,
                proposals,
                appearances_by_asset,
                window=window,
                faces_by_asset=faces_by_asset,
                rejected=rejected,
                include_pin_divergent=True,
            )
            for card in map(asdict, cards):
                # The persisted decision, if any, wins over the default.
                card["included"] = decisions.get(
                    (card["asset_id"], card["appearance"]), card["included"]
                )
                card["pinned_elsewhere"] = (
                    card["pin"] is not None and card["pin"] != card["proposed_name"]
                )
                all_cards.append(card)

        writable_by_name: dict[str, list[dict[str, Any]]] = {}
        pinned_by_name: dict[str, list[dict[str, Any]]] = {}
        labeled_by_name: dict[str, list[dict[str, Any]]] = {}
        unnamed: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for card in all_cards:
            if card["pinned_elsewhere"]:
                # Pinned to another person: shown under that person, not
                # writable until the next batch run regroups it.
                pinned_by_name.setdefault(card["pin"], []).append(card)
                continue
            if card["proposed_name"]:
                writable_by_name.setdefault(card["proposed_name"], []).append(card)
            elif card["status"] == "already_labeled" and card["immich_label"]:
                labeled_by_name.setdefault(card["immich_label"], []).append(card)
            elif card["status"] == "ask_for_name" or (
                card["status"] == "will_write" and not card["proposed_name"]
            ):
                # Nameless proposals (unnamed clusters, or a degenerate
                # proposed row) need a name before they are writable: the
                # unnamed tab hands them to the queue for naming.
                unnamed.setdefault((card["window"], card["cluster"]), []).append(card)

        tab_names = set(writable_by_name) | set(labeled_by_name) | set(pinned_by_name)
        tabs = []
        for name in sorted(tab_names):
            writable_cards = writable_by_name.get(name, [])
            labeled_cards = labeled_by_name.get(name, [])
            pinned_cards = pinned_by_name.get(name, [])
            tabs.append(
                {
                    "name": name,
                    "writable_cards": writable_cards,
                    "labeled_cards": labeled_cards,
                    "pinned_cards": pinned_cards,
                    "writable": sum(
                        1
                        for c in writable_cards
                        if c["status"] == "will_write" and c["included"]
                    ),
                    "labeled": len(labeled_cards),
                    "pinned": len(pinned_cards),
                }
            )
        unnamed_list = [
            {"window": window, "cluster": cluster, "cards": cards}
            for (window, cluster), cards in sorted(unnamed.items())
        ]
        return {
            "tabs": tabs,
            "unnamed": unnamed_list,
            "total_writable": sum(t["writable"] for t in tabs),
            "pending": pending,
            "total_pending": len(pending),
        }

    def set_write_decision(
        self, asset_id: str, appearance: int, included: bool
    ) -> dict[str, Any]:
        """Persist the operator's write/skip choice for one appearance."""
        self.store.set_write_decision(asset_id, appearance, included)
        return {
            "asset_id": asset_id,
            "appearance": appearance,
            "included": included,
        }

    def _require_idle(self) -> dict[str, Any] | None:
        """Confirmations pause while a pipeline run is rebuilding proposals."""
        if self.jobs.active():
            return {"error": "a pipeline run is in progress — confirmations are paused"}
        return None

    def confirm_all(self, body: dict[str, Any]) -> dict[str, Any]:
        """Confirm every pending write in one go, grouped by name/window/cluster.

        Dry-run records one batch per (name, window, cluster); live writes
        them all. Write decisions, rejections and ignores are respected —
        this composes the existing per-cluster confirm, nothing new writes
        outside it.
        """
        paused = self._require_idle()
        if paused:
            return paused
        dry_run = bool(body.get("dry_run", True))
        review = self.get_write_review()
        groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for tab in review["tabs"]:
            for card in tab["writable_cards"]:
                if card["status"] == "will_write" and card["included"]:
                    key = (tab["name"], card["window"], card["cluster"])
                    groups.setdefault(key, []).append(
                        {"asset_id": card["asset_id"], "appearance": card["appearance"]}
                    )
        if not groups:
            return {"error": "nothing to write"}

        results: list[dict[str, Any]] = []
        for (name, window, cluster), included in sorted(groups.items()):
            group_result = self.confirm_cluster(
                {
                    "cluster": cluster,
                    "window": window,
                    "name": name,
                    "included": included,
                    "dry_run": dry_run,
                }
            )
            results.append(
                {
                    "name": name,
                    "window": window,
                    "cluster": cluster,
                    "result": group_result,
                }
            )
        return {
            "groups": results,
            "count": sum(len(g) for g in groups.values()),
            "batch_ids": [
                r["result"]["batch_id"] for r in results if "batch_id" in r["result"]
            ],
        }

    def promote_all(self) -> dict[str, Any]:
        """Promote every un-promoted dry-run batch, one click.

        Batches whose promote reports an error are not counted as promoted
        and stay in the list for another try.
        """
        paused = self._require_idle()
        if paused:
            return paused
        results = []
        for batch in self.store.get_dry_run_batches():
            if batch.promoted_at is not None:
                continue
            outcome = self.promote_batch(batch.id)
            if "error" not in outcome:
                results.append({"batch_id": batch.id, **outcome})
        return {"promoted": len(results), "results": results}

    def unreject_appearance(
        self, asset_id: str, appearance: int, proposed_name: str
    ) -> dict[str, Any]:
        """Undo a rejection: the pair may be proposed again.

        The proposal's status is re-derived the way anchoring would (own
        Immich label, else the cluster's anchor name, else ask-for-name);
        the next batch run re-derives everything wholesale anyway.
        """
        if not proposed_name:
            return {"error": "the rejected name is required to undo a rejection"}
        self.store.remove_appearance_rejection(asset_id, appearance, proposed_name)
        status = self._rederive_status(asset_id, appearance)
        self.store.update_proposal_status(asset_id, appearance, status)
        return {
            "asset_id": asset_id,
            "appearance": appearance,
            "status": status.value,
        }

    def _rederive_status(self, asset_id: str, appearance: int) -> ProposalStatus:
        """Recompute one appearance's status the way anchoring would."""
        proposals = self.store.proposals()
        proposal = next(
            (
                p
                for p in proposals
                if p.asset_id == asset_id and p.appearance == appearance
            ),
            None,
        )
        if proposal is None:
            return ProposalStatus.PROPOSED
        appearance_row = next(
            (
                a
                for a in self.store.appearances_by_asset().get(asset_id, ())
                if a.appearance == appearance
            ),
            None,
        )
        if appearance_row is not None:
            try:
                faces = self.immich.fetch_faces(asset_id)
            except Exception:  # noqa: BLE001
                faces = []
            face = claim_face(appearance_row, faces, self.cfg.clustering.face_overlap)
            if face is not None and face.person_name:
                return ProposalStatus.LABELED
        cluster_props = [
            p
            for p in proposals
            if p.cluster == proposal.cluster and p.window == proposal.window
        ]
        proposed_name = next(
            (p.proposed_name for p in cluster_props if p.proposed_name), None
        )
        if proposed_name:
            return ProposalStatus.PROPOSED
        return ProposalStatus.ASK_NAME

    def get_proposals(self) -> list[dict[str, Any]]:
        """Return all proposals as raw dicts."""
        return [asdict(p) for p in self.store.proposals()]

    def get_summary(self) -> list[dict[str, Any]]:
        """Return window summaries from the latest run (recomputed from proposals)."""
        return [asdict(s) for s in self._build_stats()]

    def _make_write_client(self) -> ImmichClient:
        """Create a write-scoped Immich client with head detector for fallback."""
        assert self.cfg.immich.write_api_key is not None, (
            "write_api_key must be configured"
        )
        head_cfg = HeadDetectorConfig(
            weights_path=self.cfg.head_detection.weights_path,
            input_size=self.cfg.head_detection.input_size,
            confidence_threshold=self.cfg.head_detection.confidence_threshold,
            top_expansion_ratio=self.cfg.head_detection.top_expansion_ratio,
            width_expansion_ratio=self.cfg.head_detection.width_expansion_ratio,
            nms_threshold=self.cfg.head_detection.nms_threshold,
        )
        return ImmichClient(
            self.cfg.immich.base_url,
            self.cfg.immich.write_api_key,
            page_size=self.cfg.immich.page_size,
            timeout=self.cfg.immich.timeout_seconds,
            head_detector_cfg=head_cfg,
        )

    def _write_appearance(
        self,
        write_client: ImmichClient,
        asset_id: str,
        appearance: Appearance,
        person_id: str,
        cluster: int,
        window: str,
        timestamp: datetime,
        dry_run: bool = False,
        expected_head_box: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Shared write logic for one appearance (used by confirm and promote)."""
        try:
            return self._write_appearance_checked(
                write_client,
                asset_id,
                appearance,
                person_id,
                cluster,
                window,
                timestamp,
                dry_run,
                expected_head_box,
            )
        except ImmichError as exc:
            # Immich hiccups (missing pixels, transient errors) are per-item
            # failures, not confirm crashes: report and continue.
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.ERROR.value,
                "message": str(exc),
            }

    def _write_appearance_checked(
        self,
        write_client: ImmichClient,
        asset_id: str,
        appearance: Appearance,
        person_id: str,
        cluster: int,
        window: str,
        timestamp: datetime,
        dry_run: bool = False,
        expected_head_box: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        # Check if already written (via write ledger) - idempotency
        if self.store.has_write_ledger_entry(asset_id, person_id):
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.SKIPPED.value,
                "message": "already written (ledger)",
            }

        # Check tombstone
        if self.store.has_tombstone(asset_id, person_id):
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.SKIPPED.value,
                "message": "tombstone (manually removed in Immich)",
            }

        # Check existing faces for conflicts and already-labeled
        person_box = appearance.box
        has_conflict, already_labeled = write_client.check_existing_faces(
            asset_id, person_box, person_id, self.cfg.clustering.face_overlap
        )

        if already_labeled:
            # Already has this person's face - skip and journal
            if not dry_run:
                self.store.add_write_ledger_entry(
                    WriteLedgerEntry(
                        asset_id=asset_id,
                        person_id=person_id,
                        cluster=cluster,
                        window=window,
                        timestamp=timestamp,
                    )
                )
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.SKIPPED.value,
                "message": "already labeled in Immich",
            }

        if has_conflict:
            # Different person's overlapping face - conflict
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.SKIPPED.value,
                "message": "conflict: different person's overlapping face",
            }

        # Resolve head box
        if dry_run:
            head_box = write_client.resolve_head_box(
                asset_id, person_box, self.cfg.clustering.face_overlap
            )
        else:
            # Promote: use the recorded head box (replay recorded actions)
            head_box = expected_head_box
        if head_box is None:
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.SKIPPED.value,
                "message": "no head found" + (" (recorded)" if not dry_run else ""),
            }

        if dry_run:
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.WOULD_WRITE.value,
                "head_box": head_box,
            }

        # Real write
        try:
            write_client.create_face(
                asset_id,
                person_id,
                head_box[0],
                head_box[1],
                head_box[2],
                head_box[3],
            )
        except ImmichError as exc:
            return {
                "asset_id": asset_id,
                "appearance": appearance.appearance,
                "status": WriteResultStatus.ERROR.value,
                "message": f"face creation failed: {exc.message}",
            }

        # Journal the write
        self.store.add_write_ledger_entry(
            WriteLedgerEntry(
                asset_id=asset_id,
                person_id=person_id,
                cluster=cluster,
                window=window,
                timestamp=timestamp,
            )
        )

        return {
            "asset_id": asset_id,
            "appearance": appearance.appearance,
            "status": WriteResultStatus.WRITTEN.value,
            "head_box": head_box,
        }

    def _pin_divergent_props(
        self, props: list[Proposal]
    ) -> tuple[list[Proposal], list[Proposal]]:
        """Split proposals into (kept, pinned-elsewhere).

        A proposal pinned to a person other than its own proposed name has,
        in the operator's eyes, left this cluster: views exclude it until
        the next batch run regroups everything.
        """
        pins = self.store.appearance_pins()
        kept: list[Proposal] = []
        moved: list[Proposal] = []
        for prop in props:
            pin = pins.get((prop.asset_id, prop.appearance))
            if pin is not None and pin != prop.proposed_name:
                moved.append(prop)
            else:
                kept.append(prop)
        return kept, moved

    def _build_cluster_rows(
        self,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
    ) -> list[ClusterRow]:
        """Build the review queue sidebar rows grouped by window + cluster."""
        # Group by (window, cluster)
        groups: dict[tuple[str, int], list[Proposal]] = {}
        for prop in proposals:
            groups.setdefault((prop.window, prop.cluster), []).append(prop)

        # Precompute confidence lookup
        confidence_lookup = {
            (a.asset_id, a.appearance): a.confidence
            for apps in appearances_by_asset.values()
            for a in apps
        }

        rows: list[ClusterRow] = []
        dismissed = self.store.dismissed_clusters()
        for (window, cluster_id), all_props in sorted(groups.items()):
            # Proposals pinned to another person have left this cluster.
            props, _moved = self._pin_divergent_props(all_props)
            if not props:
                continue
            # Shared status counts
            writable, conflicted, no_head, labeled = self._status_counts(props)
            proposed_name, anchor = self._anchor_and_name(props)

            # Confidence = average of proposal confidences
            confidences = [
                confidence_lookup.get((p.asset_id, p.appearance), 0.0) for p in props
            ]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # A cluster where Immich labeled every appearance has no pending
            # work; the queue hides it by default (the review page's person
            # tabs remain the surface for those labels).
            fully_labeled = all(p.status == ProposalStatus.LABELED for p in props)
            if fully_labeled:
                proposed_name, anchor = None, "already labeled in Immich"

            rows.append(
                ClusterRow(
                    cluster=cluster_id,
                    window=window,
                    proposed_name=proposed_name,
                    anchor=anchor,
                    confidence=confidence,
                    appearances=len(props),
                    writable=writable,
                    conflicted=conflicted,
                    ask_for_name=no_head,
                    labeled=labeled,
                    # Operator dismissal persists; everything else is open.
                    status=(
                        ClusterRowStatus.SKIPPED
                        if (window, cluster_id) in dismissed
                        else ClusterRowStatus.OPEN
                    ),
                    # First few appearance refs; the UI fetches each as a
                    # person crop via /api/crop.
                    crops=[
                        {"asset_id": p.asset_id, "appearance": p.appearance}
                        for p in props[:4]
                    ],
                    fully_labeled=fully_labeled,
                )
            )
        return rows

    def _status_counts(self, props: list[Proposal]) -> tuple[int, int, int, int]:
        """Return (writable, conflicted, no_head, labeled) counts for a proposal group."""
        return (
            sum(1 for p in props if p.status == ProposalStatus.PROPOSED),
            sum(1 for p in props if p.status == ProposalStatus.CONFLICTED),
            sum(
                1 for p in props if p.status == ProposalStatus.ASK_NAME
            ),  # ask-for-name; real no-head in #14
            sum(1 for p in props if p.status == ProposalStatus.LABELED),
        )

    def _anchor_and_name(self, props: list[Proposal]) -> tuple[str | None, str | None]:
        """Return (proposed_name, anchor_description) for a proposal group."""
        proposed_name = next((p.proposed_name for p in props if p.proposed_name), None)
        if proposed_name:
            return proposed_name, f"Immich label: {proposed_name}"
        if any(p.status == ProposalStatus.ASK_NAME for p in props):
            return None, "needs a name"
        return None, None

    def _asset_name(self, asset_id: str) -> str:
        """Human-readable file name, fetched once and cached per process."""
        if asset_id not in self._asset_names:
            try:
                name = self.immich.fetch_asset_name(asset_id)
            except Exception:  # noqa: BLE001
                name = None
            self._asset_names[asset_id] = name or f"{asset_id[:8]}…"
        return self._asset_names[asset_id]

    def _build_evidence_for_cluster(
        self,
        cluster: int | None,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
        window: str | None = None,
        *,
        faces_by_asset: dict[str, list[Face]] | None = None,
        rejected: set[tuple[str, int, str]] | None = None,
        include_pin_divergent: bool = False,
    ) -> list[EvidenceCard]:
        """Build evidence cards for one cluster.

        ``faces_by_asset`` carries Immich's face detections so cards can show
        the appearance's current Immich label; ``rejected`` holds operator
        wrong-label flags, which blank the label out.
        """
        if cluster is None:
            return []

        faces_by_asset = faces_by_asset or {}
        rejected = rejected or set()
        # Operator pins: shown on the card; the batch run applies them.
        pins = self.store.appearance_pins()
        # Operator ignores: "this detection is wrong"; the batch run drops them.
        ignored = self.store.appearance_ignores()

        target_props = [
            p
            for p in proposals
            if p.cluster == cluster and (window is None or p.window == window)
        ]
        if not include_pin_divergent:
            # Proposals pinned to another person have left this cluster: the
            # review page requests them explicitly to show them under the
            # pinned person instead.
            target_props, _moved = self._pin_divergent_props(target_props)
        if not target_props:
            return []

        # The cluster's distinct claimed names (anchoring's rule: best claim
        # per appearance, rejected faces excluded). Two or more mean the
        # cluster is conflicted; the cards say which labels are fighting.
        cluster_names: set[str] = set()
        for prop in target_props:
            appearance = next(
                (
                    a
                    for a in appearances_by_asset.get(prop.asset_id, [])
                    if a.appearance == prop.appearance
                ),
                None,
            )
            if appearance is None:
                continue
            faces = [
                f
                for f in faces_by_asset.get(prop.asset_id, ())
                if not (
                    f.person_name
                    and (prop.asset_id, prop.appearance, f.person_name) in rejected
                )
            ]
            face = claim_face(appearance, faces, self.cfg.clustering.face_overlap)
            if face is not None and face.person_name:
                cluster_names.add(face.person_name)
        conflict_names = sorted(cluster_names) if len(cluster_names) >= 2 else None

        cards: list[EvidenceCard] = []
        for prop in target_props:
            app_list = appearances_by_asset.get(prop.asset_id, [])
            appearance = next(
                (a for a in app_list if a.appearance == prop.appearance), None
            )
            if appearance is None:
                continue

            # The Immich face claiming this appearance, if any: the label the
            # user sees. Same containment rule the anchoring uses.
            immich_label: str | None = None
            best_ratio = 0.0
            for face in faces_by_asset.get(prop.asset_id, ()):
                ratio = face_person_overlap(appearance, face)
                if (
                    ratio >= self.cfg.clustering.face_overlap
                    and ratio > best_ratio
                    and face.person_name
                ):
                    best_ratio = ratio
                    immich_label = face.person_name
            if (
                immich_label
                and (
                    prop.asset_id,
                    prop.appearance,
                    immich_label,
                )
                in rejected
            ):
                immich_label = None  # operator flagged this label as wrong

            # Status chip mapping
            status_map = {
                ProposalStatus.PROPOSED: EvidenceCardStatus.WILL_WRITE,
                ProposalStatus.LABELED: EvidenceCardStatus.ALREADY_LABELED,
                ProposalStatus.CONFLICTED: EvidenceCardStatus.CONFLICT_SKIP,
                ProposalStatus.ASK_NAME: EvidenceCardStatus.ASK_NAME,
                ProposalStatus.REJECTED: EvidenceCardStatus.REJECTED,
            }
            status = status_map.get(prop.status, EvidenceCardStatus.WILL_WRITE)
            included = (
                prop.status == ProposalStatus.PROPOSED
            )  # default include writable

            # Date from captured_at - convert to ISO string for JSON serialization
            date_str = (
                isoformat_z(appearance.captured_at)
                if appearance.captured_at is not None
                else "unknown"
            )

            cards.append(
                EvidenceCard(
                    asset_id=prop.asset_id,
                    appearance=prop.appearance,
                    window=prop.window,
                    cluster=prop.cluster,
                    date=date_str,
                    confidence=appearance.confidence,
                    immich_label=immich_label,
                    status=status,
                    included=included,
                    pixel_source=appearance.pixel_source,
                    pin=pins.get((prop.asset_id, prop.appearance)),
                    conflict_names=(
                        conflict_names
                        if status == EvidenceCardStatus.CONFLICT_SKIP
                        else None
                    ),
                    asset_name=self._asset_name(prop.asset_id),
                    ignored=(prop.asset_id, prop.appearance) in ignored,
                    proposed_name=prop.proposed_name,
                )
            )
        return cards

    def _build_assignment(
        self,
        cluster: int | None,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
        window: str | None = None,
    ) -> AssignmentPane | None:
        """Build the right assignment pane for one cluster in one window."""
        if cluster is None:
            return None

        target_props = [
            p
            for p in proposals
            if p.cluster == cluster and (window is None or p.window == window)
        ]
        if not target_props:
            return None

        # Proposals pinned to another person have left this cluster.
        target_props, _moved = self._pin_divergent_props(target_props)
        if not target_props:
            return None

        window = target_props[0].window
        proposed_name, anchor = self._anchor_and_name(target_props)
        writable, conflicted, ask_for_name, labeled = self._status_counts(target_props)
        return AssignmentPane(
            cluster=cluster,
            window=window,
            proposed_name=proposed_name,
            anchor=anchor,
            total_appearances=len(target_props),
            writable=writable,
            conflicted=conflicted,
            ask_for_name=ask_for_name,
            labeled=labeled,
            existing_people=self.get_people(),
        )

    def _build_stats(self) -> list[StatsPane]:
        """Rebuild window summaries from stored proposals."""
        proposals = self.store.proposals()
        windows: dict[str, list[Proposal]] = {}
        for p in proposals:
            windows.setdefault(p.window, []).append(p)

        stats: list[StatsPane] = []
        for window, props in sorted(windows.items()):
            # Count unique clusters (multi-appearance only)
            clusters_by_id: dict[int, list[Proposal]] = {}
            for p in props:
                clusters_by_id.setdefault(p.cluster, []).append(p)

            multi = {cid: mem for cid, mem in clusters_by_id.items() if len(mem) >= 2}
            clusters = len(multi)
            anchored = sum(
                1 for mem in multi.values() if any(p.proposed_name for p in mem)
            )
            conflicted = sum(
                1
                for mem in multi.values()
                if any(p.status == ProposalStatus.CONFLICTED for p in mem)
            )
            anchor_less = sum(
                1
                for mem in multi.values()
                if all(p.status == ProposalStatus.ASK_NAME for p in mem)
            )
            singletons = sum(1 for mem in clusters_by_id.values() if len(mem) == 1)
            proposals_count = sum(
                1 for p in props if p.status == ProposalStatus.PROPOSED
            )
            ask_for_name = sum(1 for p in props if p.status == ProposalStatus.ASK_NAME)

            def pct(n: int, clusters: int = clusters) -> float:
                return 100.0 * n / clusters if clusters else 0.0

            stats.append(
                StatsPane(
                    window=window,
                    appearances=len(props),
                    clusters=clusters,
                    anchored_pct=pct(anchored),
                    conflicted_pct=pct(conflicted),
                    anchor_less_pct=pct(anchor_less),
                    singletons=singletons,
                    proposals=proposals_count,
                    ask_for_name=ask_for_name,
                )
            )
        return stats

    def confirm_cluster(self, body: dict[str, Any]) -> dict[str, Any]:
        """Confirm a cluster: write head boxes to Immich for included appearances.

        The body should contain:
        - cluster: int (cluster index)
        - window: str (window key)
        - name: str (person name, existing or new)
        - included: list[dict] with asset_id, appearance for included appearances
        - dry_run: bool (whether this is a dry-run)

        Returns a dict with the results of the confirmation.
        """
        paused = self._require_idle()
        if paused:
            return paused
        cluster = int(body.get("cluster", -1))
        window = str(body.get("window", ""))
        name = str(body.get("name", "")).strip()
        included = body.get("included", [])

        if cluster < 0 or not window or not name:
            return {"error": "missing required fields"}

        if not included:
            return {"error": "no appearances included"}

        # Get the proposals for this cluster. Anchored clusters propose
        # (PROPOSED); ask-for-name clusters become writable once the operator
        # supplies the name — that is the ask-for-name feature.
        writable_statuses = {ProposalStatus.PROPOSED, ProposalStatus.ASK_NAME}
        proposals = self.store.proposals()
        target_props = [
            p
            for p in proposals
            if p.cluster == cluster
            and p.window == window
            and p.status in writable_statuses
        ]

        if not target_props:
            return {"error": "no writable proposals found for this cluster"}

        # Operator rejections: an appearance marked "not this person" for this
        # name is never written, and the pair is skipped with a clear reason.
        rejected = {
            (asset_id, appearance)
            for asset_id, appearance, rejected_name in self.store.appearance_rejections()
            if rejected_name == name
        }
        # Operator ignores: "this detection is wrong" skips the write even in
        # the same session, before the next run drops it from the queue.
        ignored = self.store.appearance_ignores()

        # Build a set of included (asset_id, appearance) for quick lookup
        included_set = {
            (item["asset_id"], item["appearance"])
            for item in included
            if "asset_id" in item and "appearance" in item
        }

        # Filter to only included proposals
        write_props = [
            p
            for p in target_props
            if (p.asset_id, p.appearance) in included_set
            and (p.asset_id, p.appearance) not in rejected
            and (p.asset_id, p.appearance) not in ignored
        ]

        if not write_props:
            return {"error": "none of the included appearances are writable"}

        # Resolve the target person (for dry-run we still need the real person_id
        # to check ledger/tombstones correctly)
        existing_person = self.immich.find_person_by_name(name)
        if existing_person is not None:
            person_id = existing_person.id
        else:
            # No existing person; use a placeholder that won't match any real ledger entry
            # The shared helper uses person_id for ledger/tombstone checks, so this
            # ensures dry-run correctly shows "would_write" for new people
            person_id = f"new-person:{name}"

        results = []
        dry_run_actions = []
        timestamp = datetime.now(UTC)

        for prop in write_props:
            appearances_by_asset = self.store.appearances_by_asset()
            app_list = appearances_by_asset.get(prop.asset_id, [])
            appearance = next(
                (a for a in app_list if a.appearance == prop.appearance), None
            )
            if appearance is None:
                result = {
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": WriteResultStatus.ERROR.value,
                    "message": "appearance not found",
                }
                results.append(result)
                dry_run_actions.append(
                    DryRunAction(
                        asset_id=prop.asset_id,
                        appearance=prop.appearance,
                        head_box=None,
                        status=DryRunActionStatus.ERROR,
                        message="appearance not found",
                    )
                )
                continue

            # Use shared write logic (dry_run=True)
            result = self._write_appearance(
                write_client=self.immich,  # read client has head detector
                asset_id=prop.asset_id,
                appearance=appearance,
                person_id=person_id,
                cluster=cluster,
                window=window,
                timestamp=timestamp,
                dry_run=True,
            )
            results.append(result)
            dry_run_actions.append(
                DryRunAction(
                    asset_id=prop.asset_id,
                    appearance=prop.appearance,
                    head_box=result.get("head_box"),  # type: ignore[arg-type]
                    status=DryRunActionStatus(result["status"]),
                    message=result.get("message"),  # type: ignore[arg-type]
                )
            )

        # Persist the dry-run batch
        batch_id = self.store.add_dry_run_batch(
            name, window, cluster, timestamp, dry_run_actions
        )

        return {
            "cluster": cluster,
            "window": window,
            "name": name,
            "dry_run": True,
            "batch_id": batch_id,
            "results": results,
        }

        # Real write path: need write-scoped API key
        write_api_key = self.cfg.immich.write_api_key
        if not write_api_key:
            return {"error": "write-scoped API key not configured"}

        # Create a write-scoped client (with head detector for fallback)
        write_client = self._make_write_client()

        # Find or create the person
        existing_person = write_client.find_person_by_name(name)
        if existing_person is not None:
            person_id = existing_person.id
        else:
            try:
                created = write_client.create_person(name)
                person_id = created.id
            except ImmichError as exc:
                return {"error": f"failed to create person: {exc.message}"}

        results = []
        timestamp = datetime.now(UTC)

        for prop in write_props:
            appearances_by_asset = self.store.appearances_by_asset()
            app_list = appearances_by_asset.get(prop.asset_id, [])
            appearance = next(
                (a for a in app_list if a.appearance == prop.appearance), None
            )
            if appearance is None:
                results.append(
                    {
                        "asset_id": prop.asset_id,
                        "appearance": prop.appearance,
                        "status": WriteResultStatus.ERROR.value,
                        "message": "appearance not found",
                    }
                )
                continue

            # Use shared write logic (dry_run=False)
            result = self._write_appearance(
                write_client=write_client,
                asset_id=prop.asset_id,
                appearance=appearance,
                person_id=person_id,
                cluster=cluster,
                window=window,
                timestamp=timestamp,
                dry_run=False,
            )
            results.append(result)

        return {
            "cluster": cluster,
            "window": window,
            "name": name,
            "person_id": person_id,
            "dry_run": False,
            "results": results,
        }

    def promote_batch(self, batch_id: str) -> dict[str, Any]:
        """Promote a dry-run batch: replay its actions as real writes to Immich.

        Idempotent: promoting an already-promoted batch returns success with
        promoted: true and the original person_id.
        """
        paused = self._require_idle()
        if paused:
            return paused

        # Get the dry-run batch
        batch = self.store.get_dry_run_batch(batch_id)
        if batch is None:
            return {"error": f"batch not found: {batch_id}"}

        if batch.promoted_at is not None:
            # Already promoted - return success with the original person_id for idempotency
            return {
                "batch_id": batch_id,
                "promoted": True,
                "person_id": batch.person_id or batch.name,
                "name": batch.name,
                "window": batch.window,
                "cluster": batch.cluster,
                "already_promoted": True,
            }

        # Need write-scoped API key
        write_api_key = self.cfg.immich.write_api_key
        if not write_api_key:
            return {"error": "write-scoped API key not configured"}

        # Create a write-scoped client (with head detector for fallback)
        write_client = self._make_write_client()

        # Find or create the person
        existing_person = write_client.find_person_by_name(batch.name)
        if existing_person is not None:
            person_id = existing_person.id
        else:
            try:
                created = write_client.create_person(batch.name)
                person_id = created.id
            except ImmichError as exc:
                return {"error": f"failed to create person: {exc.message}"}

        results = []
        timestamp = datetime.now(UTC)

        for action in batch.actions:
            # Only replay actions that would_write
            if action.status != DryRunActionStatus.WOULD_WRITE:
                results.append(
                    {
                        "asset_id": action.asset_id,
                        "appearance": action.appearance,
                        "status": WriteResultStatus.SKIPPED.value,
                        "message": action.message or "not a writable action",
                    }
                )
                continue

            # Get the appearance for the person box
            appearances_by_asset = self.store.appearances_by_asset()
            app_list = appearances_by_asset.get(action.asset_id, [])
            appearance = next(
                (a for a in app_list if a.appearance == action.appearance), None
            )
            if appearance is None:
                results.append(
                    {
                        "asset_id": action.asset_id,
                        "appearance": action.appearance,
                        "status": WriteResultStatus.ERROR.value,
                        "message": "appearance not found",
                    }
                )
                continue

            # Use shared write logic (dry_run=False, replay recorded head box)
            result = self._write_appearance(
                write_client=write_client,
                asset_id=action.asset_id,
                appearance=appearance,
                person_id=person_id,
                cluster=batch.cluster,
                window=batch.window,
                timestamp=timestamp,
                dry_run=False,
                expected_head_box=action.head_box,
            )
            results.append(result)
            if result["status"] == WriteResultStatus.WRITTEN.value:
                self.store.mark_dry_run_action_written(
                    batch_id, action.asset_id, action.appearance, timestamp
                )

        # Mark the batch as promoted with the person_id
        self.store.mark_dry_run_batch_promoted(batch_id, timestamp, person_id)

        return {
            "batch_id": batch_id,
            "promoted": True,
            "person_id": person_id,
            "name": batch.name,
            "window": batch.window,
            "cluster": batch.cluster,
            "results": results,
        }

    def write_pending(self, body: dict[str, Any]) -> dict[str, Any]:
        """Write exactly the selected pending items, one click on the review page.

        Each item identifies an action of an un-promoted dry-run batch. The
        write reuses the batch promote path; every written action is marked,
        and a batch whose actions are all written closes automatically —
        unselected items stay pending for a later pass.
        """
        paused = self._require_idle()
        if paused:
            return paused
        items = body.get("items", [])
        if not items:
            return {"error": "no items selected"}

        by_batch: dict[str, list[tuple[str, int]]] = {}
        for item in items:
            batch_id = str(item.get("batch_id", ""))
            asset_id = str(item.get("asset_id", ""))
            try:
                appearance = int(item.get("appearance", -1))
            except (TypeError, ValueError):
                return {"error": "invalid item in selection"}
            if not batch_id or not asset_id:
                return {"error": "invalid item in selection"}
            by_batch.setdefault(batch_id, []).append((asset_id, appearance))

        results: list[dict[str, Any]] = []
        timestamp = datetime.now(UTC)
        for batch_id, wanted in by_batch.items():
            batch = self.store.get_dry_run_batch(batch_id)
            if batch is None:
                results.append(
                    {"batch_id": batch_id, "error": f"batch not found: {batch_id}"}
                )
                continue
            if batch.promoted_at is not None:
                results.append({"batch_id": batch_id, "error": "batch already written"})
                continue

            write_api_key = self.cfg.immich.write_api_key
            if not write_api_key:
                return {"error": "write-scoped API key not configured"}
            write_client = self._make_write_client()
            existing_person = write_client.find_person_by_name(batch.name)
            if existing_person is not None:
                person_id = existing_person.id
            else:
                try:
                    person_id = write_client.create_person(batch.name).id
                except ImmichError as exc:
                    results.append(
                        {
                            "batch_id": batch_id,
                            "error": f"failed to create person: {exc.message}",
                        }
                    )
                    continue

            wanted_set = set(wanted)
            appearances_by_asset = self.store.appearances_by_asset()
            for action in batch.actions:
                if (action.asset_id, action.appearance) not in wanted_set:
                    continue  # stays pending for a later pass
                if action.written_at is not None:
                    continue  # already written
                app_list = appearances_by_asset.get(action.asset_id, [])
                appearance_row = next(
                    (a for a in app_list if a.appearance == action.appearance),
                    None,
                )
                if (
                    action.status != DryRunActionStatus.WOULD_WRITE
                    or appearance_row is None
                ):
                    result = {
                        "asset_id": action.asset_id,
                        "appearance": action.appearance,
                        "status": WriteResultStatus.SKIPPED.value,
                        "message": action.message or "not a writable action",
                    }
                else:
                    result = self._write_appearance(
                        write_client=write_client,
                        asset_id=action.asset_id,
                        appearance=appearance_row,
                        person_id=person_id,
                        cluster=batch.cluster,
                        window=batch.window,
                        timestamp=timestamp,
                        dry_run=False,
                        expected_head_box=action.head_box,
                    )
                results.append({"batch_id": batch_id, **result})
                if result["status"] == WriteResultStatus.WRITTEN.value:
                    self.store.mark_dry_run_action_written(
                        batch_id, action.asset_id, action.appearance, timestamp
                    )
            # The batch closes once nothing in it is left un-written.
            self.store.close_dry_run_batch_if_complete(batch_id, timestamp)

        written = sum(
            1 for r in results if r.get("status") == WriteResultStatus.WRITTEN.value
        )
        return {"count": len(items), "written": written, "results": results}

    def get_dry_run_batches(self) -> list[dict[str, Any]]:
        """Return all persisted dry-run batches as dicts."""
        batches = self.store.get_dry_run_batches()
        return [
            {
                "id": batch.id,
                "name": batch.name,
                "window": batch.window,
                "cluster": batch.cluster,
                "timestamp": batch.timestamp.isoformat(),
                "promoted_at": batch.promoted_at.isoformat()
                if batch.promoted_at
                else None,
                "actions": [
                    {
                        "asset_id": action.asset_id,
                        "appearance": action.appearance,
                        "head_box": (
                            [float(v) for v in action.head_box]
                            if action.head_box is not None
                            else None
                        ),
                        "status": action.status.value,
                        "message": action.message,
                    }
                    for action in batch.actions
                ],
            }
            for batch in batches
        ]


def serve_core(
    config_path: Path, host: str | None = None, port: int | None = None
) -> None:
    """Run the core API server.

    The bind address comes from [core] host/port unless overridden here; the
    code default is loopback, and opting into a wider bind is an explicit
    config or CLI choice (it also needs an api_key).
    """
    cfg = load_config(config_path)
    host = cfg.core.host if host is None else host
    port = cfg.core.port if port is None else port
    core_server = CoreServer(cfg, config_path)

    CoreAPIHandler.core_server = core_server

    server = HTTPServer((host, port), CoreAPIHandler)
    _log.info("core API server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _log.info("core API server stopped")


def main() -> None:
    """Entry point for `reid-core`."""
    import click

    @click.command()
    @click.option(
        "--config",
        "config_path",
        required=True,
        type=click.Path(path_type=Path, dir_okay=False),
    )
    @click.option(
        "--host",
        default=None,
        help="Bind address (default: [core] host, loopback)",
    )
    @click.option(
        "--port",
        default=None,
        type=int,
        help="Port to listen on (default: [core] port, 8090)",
    )
    def core(config_path: Path, host: str | None, port: int | None) -> None:
        """Run the core API server."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
        serve_core(config_path, host, port)

    core()


if __name__ == "__main__":
    main()
