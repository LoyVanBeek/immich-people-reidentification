"""The confirmation UI HTTP API server.

A small read-only HTTP API that serves the latest run's proposals,
evidence, and stats to the confirmation workspace. No Immich writes
occur from this API — it only reads from the local store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .anchoring import face_person_overlap
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

_log = logging.getLogger("reid.server")

# Static file directory (served at /)
STATIC_DIR = Path(__file__).parent / "ui"

_UI_VERSION: str | None = None


def ui_version() -> str:
    """Stable content hash of the static assets, for cache-busting URLs."""
    global _UI_VERSION
    if _UI_VERSION is None:
        digest = hashlib.sha256()
        for name in ("app.js", "style.css"):
            digest.update((STATIC_DIR / name).read_bytes())
        _UI_VERSION = digest.hexdigest()[:12]
    return _UI_VERSION


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


class UIAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the confirmation UI API."""

    # Class variable set by serve()
    ui_server: UIServer | None = None

    def do_GET(self) -> None:
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
            self._send_json(self.ui_server.get_ui_state())  # type: ignore[union-attr]
        elif path == "/api/evidence":
            self._send_json(self.ui_server.get_evidence(parse_qs(parsed.query)))  # type: ignore[union-attr]
        elif path == "/api/people":
            self._send_json({"people": self.ui_server.get_people()})  # type: ignore[union-attr]
        elif path == "/api/config":
            self._send_json(asdict(self.ui_server.get_config_knobs()))  # type: ignore[union-attr]
        elif path == "/api/proposals":
            self._send_json(self.ui_server.get_proposals())  # type: ignore[union-attr]
        elif path == "/api/summary":
            self._send_json(self.ui_server.get_summary())  # type: ignore[union-attr]
        elif path == "/api/dry-run-batches":
            self._send_json(self.ui_server.get_dry_run_batches())  # type: ignore[union-attr]
        elif path == "/api/crop":
            try:
                data, ctype = self.ui_server.get_crop(parse_qs(parsed.query))  # type: ignore[union-attr]
            except (ValueError, LookupError) as exc:
                self.send_error(404, str(exc))
                return
            self._send_bytes(data, ctype)
        elif path == "/" or path == "/index.html":
            self._serve_index()
        else:
            self._serve_static(path.lstrip("/"))

    def do_POST(self) -> None:
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
            result = self.ui_server.confirm_cluster(body)  # type: ignore[union-attr]
            self._send_json(result)
        elif path == "/api/skip":
            body = self._read_json()
            if body is None:
                return
            result = self.ui_server.dismiss_cluster(  # type: ignore[union-attr]
                str(body.get("window", "")),
                int(body.get("cluster", -1)),
                bool(body.get("dismissed", True)),
            )
            self._send_json(result)
        elif path == "/api/reject":
            body = self._read_json()
            if body is None:
                return
            result = self.ui_server.reject_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                str(body.get("name", "")).strip(),
            )
            self._send_json(result)
        elif path == "/api/pin":
            body = self._read_json()
            if body is None:
                return
            result = self.ui_server.pin_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
                str(body.get("name", "")).strip(),
            )
            self._send_json(result)
        elif path == "/api/unpin":
            body = self._read_json()
            if body is None:
                return
            result = self.ui_server.unpin_appearance(  # type: ignore[union-attr]
                str(body.get("asset_id", "")),
                int(body.get("appearance", -1)),
            )
            self._send_json(result)
        elif path == "/api/config":
            body = self._read_json()
            if body is None:
                return
            self._send_json(self.ui_server.save_config_knobs(body))  # type: ignore[union-attr]
        elif path.startswith("/api/promote/"):
            batch_id = path[len("/api/promote/") :]
            if not batch_id:
                self.send_error(400, "Missing batch ID")
                return
            result = self.ui_server.promote_batch(batch_id)  # type: ignore[union-attr]
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
        # The UI ships as a handful of small files with no cache-busting in
        # the URLs: letting the browser hold stale copies means a deploy
        # looks like it did nothing. Always revalidate.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as fh:
            self.wfile.write(fh.read())


class UIServer:
    """The confirmation UI API server."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.store = Store(cfg.store_path)
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
        """
        import cv2

        asset_id = (query.get("asset_id") or [""])[0]
        appearance_raw = (query.get("appearance") or [""])[0]
        if not asset_id or not appearance_raw:
            raise ValueError("asset_id and appearance are required")
        appearance_no = int(appearance_raw)
        expand = float((query.get("expand") or ["0"])[0] or 0)
        outline = (query.get("outline") or ["0"])[0].lower() not in ("0", "", "false")

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

        if outline:
            # The person's box, translated into crop coordinates.
            bx0, by0 = max(left - x0, 0), max(top - y0, 0)
            bx1 = min(right - x0, x1 - x0 - 1)
            by1 = min(bottom - y0, y1 - y0 - 1)
            thickness = max(2, min(crop.shape[0], crop.shape[1]) // 80)
            cv2.rectangle(crop, (bx0, by0), (bx1, by1), (255, 140, 79), thickness)

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
        for (window, cluster_id), props in sorted(groups.items()):
            # Shared status counts
            writable, conflicted, no_head, labeled = self._status_counts(props)
            proposed_name, anchor = self._anchor_and_name(props)

            # Confidence = average of proposal confidences
            confidences = [
                confidence_lookup.get((p.asset_id, p.appearance), 0.0) for p in props
            ]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

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
            return proposed_name, f"{proposed_name} (anchored)"
        if any(p.status == ProposalStatus.ASK_NAME for p in props):
            return None, "ask for name"
        return None, None

    def _build_evidence_for_cluster(
        self,
        cluster: int | None,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
        window: str | None = None,
        *,
        faces_by_asset: dict[str, list[Face]] | None = None,
        rejected: set[tuple[str, int, str]] | None = None,
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

        target_props = [
            p
            for p in proposals
            if p.cluster == cluster and (window is None or p.window == window)
        ]
        if not target_props:
            return []

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
                        "head_box": action.head_box,
                        "status": action.status.value,
                        "message": action.message,
                    }
                    for action in batch.actions
                ],
            }
            for batch in batches
        ]


def serve(config_path: Path, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the confirmation UI HTTP server."""
    cfg = load_config(config_path)
    ui_server = UIServer(cfg)

    # Set the UIServer on the handler class
    UIAPIHandler.ui_server = ui_server

    server = HTTPServer((host, port), UIAPIHandler)
    _log.info("confirmation UI server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _log.info("confirmation UI server stopped")


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
        """Run the confirmation UI HTTP server."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
        serve(config_path, host, port)

    ui()


if __name__ == "__main__":
    main()
