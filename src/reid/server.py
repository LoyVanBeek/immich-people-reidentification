"""The confirmation UI HTTP API server.

A small read-only HTTP API that serves the latest run's proposals,
evidence, and stats to the confirmation workspace. No Immich writes
occur from this API — it only reads from the local store.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from .config import Config, load_config, ConfigError
from .head_detector import HeadDetectorConfig
from .immich import ImmichClient, ImmichError
from .store import Store, Proposal, Appearance, NEW_ASSET_FEED, WriteLedgerEntry, Tombstone
from .anchoring import WindowSummary
from .batch import run_batch

_log = logging.getLogger("reid.server")

# Static file directory (served at /)
STATIC_DIR = Path(__file__).parent / "ui"


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
    status: str  # open | confirmed | skipped
    crops: list[dict[str, Any]]  # placeholder crop data for sidebar strip


@dataclass(frozen=True)
class EvidenceCard:
    """One appearance card in the evidence grid."""

    asset_id: str
    appearance: int
    window: str
    cluster: int
    date: str
    confidence: float
    immich_label: str | None
    status: str  # will_write | already_labeled | conflict_skip | no_head_skip
    included: bool  # checkbox state for writable appearances
    pixel_source: str  # "preview" | "original"


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
    ui_server: "UIServer" = None  # type: ignore[assignment]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._send_json(self.ui_server.get_ui_state())
        elif path == "/api/evidence":
            self._send_json(self.ui_server.get_evidence(parse_qs(parsed.query)))
        elif path == "/api/people":
            self._send_json({"people": self.ui_server.get_people()})
        elif path == "/api/config":
            self._send_json(asdict(self.ui_server.get_config_knobs()))
        elif path == "/api/proposals":
            self._send_json(self.ui_server.get_proposals())
        elif path == "/api/summary":
            self._send_json(self.ui_server.get_summary())
        elif path == "/" or path == "/index.html":
            self._serve_static("index.html")
        else:
            self._serve_static(path.lstrip("/"))

    def do_POST(self) -> None:
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
            result = self.ui_server.confirm_cluster(body)
            self._send_json(result)
        else:
            self.send_error(405, "Method not allowed")

    def log_message(self, fmt: str, *args: Any) -> None:
        _log.debug(fmt % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

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
        checkpoint = self.store.get_checkpoint(NEW_ASSET_FEED)

        # Group proposals by window + cluster
        clusters = self._build_cluster_rows(proposals, appearances_by_asset)
        active_cluster = clusters[0].cluster if clusters else None
        evidence = self._build_evidence_for_cluster(active_cluster, proposals, appearances_by_asset) if active_cluster is not None else []
        assignment = self._build_assignment(active_cluster, proposals, appearances_by_asset) if active_cluster is not None else None
        stats = self._build_stats()
        config_knobs = self.get_config_knobs()
        people = self.get_people()

        state = UIState(
            clusters=clusters,
            active_cluster=active_cluster,
            evidence=evidence,
            assignment=assignment,
            stats=stats,
            config_knobs=config_knobs,
            people=people,
            dry_run=self._dry_run,
        )
        return asdict(state)

    def get_evidence(self, query: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Return evidence cards for a specific cluster."""
        cluster_q = query.get("cluster", [""])[0]
        window_q = query.get("window", [""])[0]
        if not cluster_q or not window_q:
            return []
        cluster = int(cluster_q)
        proposals = self.store.proposals()
        appearances_by_asset = self.store.appearances_by_asset()
        evidence = self._build_evidence_for_cluster(cluster, proposals, appearances_by_asset, window_q)
        return [asdict(e) for e in evidence]

    def get_people(self) -> list[str]:
        """Return sorted list of existing Immich person names for autocomplete."""
        try:
            people = self.immich.fetch_people()
            return sorted(p.name for p in people)
        except ImmichError as exc:
            _log.warning("failed to fetch people: %s", exc)
            return []

    def get_config_knobs(self) -> UIConfigKnobs:
        """Return the five tuning knobs exposed in the ⚙ dropdown."""
        return UIConfigKnobs(
            window_days=self.cfg.clustering.window_days,
            distance_threshold=self.cfg.clustering.distance_threshold,
            confidence_floor=self.cfg.detection.confidence_floor,
            min_box_height=self.cfg.detection.min_box_height,
            face_overlap=self.cfg.clustering.face_overlap,
        )

    def get_proposals(self) -> list[dict[str, Any]]:
        """Return all proposals as raw dicts."""
        return [asdict(p) for p in self.store.proposals()]

    def get_summary(self) -> list[dict[str, Any]]:
        """Return window summaries from the latest run (recomputed from proposals)."""
        return [asdict(s) for s in self._build_stats()]

    def _build_cluster_rows(
        self, proposals: list[Proposal], appearances_by_asset: dict[str, list[Appearance]]
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
        for (window, cluster_id), props in sorted(groups.items()):
            # Shared status counts
            writable, conflicted, no_head, labeled = self._status_counts(props)
            proposed_name, anchor = self._anchor_and_name(props)

            # Confidence = average of proposal confidences
            confidences = [
                confidence_lookup.get((p.asset_id, p.appearance), 0.0)
                for p in props
            ]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            rows.append(ClusterRow(
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
                status="open",  # UI-only; not persisted
                crops=[],  # placeholder; real crop thumbnails would need separate endpoint
            ))
        return rows

    def _status_counts(self, props: list[Proposal]) -> tuple[int, int, int, int]:
        """Return (writable, conflicted, no_head, labeled) counts for a proposal group."""
        return (
            sum(1 for p in props if p.status == "proposed"),
            sum(1 for p in props if p.status == "conflicted"),
            sum(1 for p in props if p.status == "ask_name"),  # ask-for-name; real no-head in #14
            sum(1 for p in props if p.status == "labeled"),
        )

    def _anchor_and_name(self, props: list[Proposal]) -> tuple[str | None, str | None]:
        """Return (proposed_name, anchor_description) for a proposal group."""
        proposed_name = next((p.proposed_name for p in props if p.proposed_name), None)
        if proposed_name:
            return proposed_name, f"{proposed_name} (anchored)"
        if any(p.status == "ask_name" for p in props):
            return None, "ask for name"
        return None, None

    def _build_evidence_for_cluster(
        self,
        cluster: int | None,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
        window: str | None = None,
    ) -> list[EvidenceCard]:
        """Build evidence cards for one cluster."""
        if cluster is None:
            return []

        target_props = [p for p in proposals if p.cluster == cluster and (window is None or p.window == window)]
        if not target_props:
            return []

        cards: list[EvidenceCard] = []
        for prop in target_props:
            app_list = appearances_by_asset.get(prop.asset_id, [])
            appearance = next((a for a in app_list if a.appearance == prop.appearance), None)
            if appearance is None:
                continue

            # Status chip mapping
            status_map = {
                "proposed": "will_write",
                "labeled": "already_labeled",
                "conflicted": "conflict_skip",
                "ask_name": "ask_for_name",  # ask-for-name queue; no-head fallback in #14
            }
            status = status_map.get(prop.status, "will_write")
            included = prop.status == "proposed"  # default include writable

            # Date from captured_at or created_at fallback
            date_str = appearance.captured_at or "unknown"

            cards.append(EvidenceCard(
                asset_id=prop.asset_id,
                appearance=prop.appearance,
                window=prop.window,
                cluster=prop.cluster,
                date=date_str,
                confidence=appearance.confidence,
                immich_label=None,  # would need to re-fetch faces to get this
                status=status,
                included=included,
                pixel_source=appearance.pixel_source,
            ))
        return cards

    def _build_assignment(
        self,
        cluster: int | None,
        proposals: list[Proposal],
        appearances_by_asset: dict[str, list[Appearance]],
    ) -> AssignmentPane | None:
        """Build the right assignment pane for a cluster."""
        if cluster is None:
            return None

        target_props = [p for p in proposals if p.cluster == cluster]
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
            anchored = sum(1 for mem in multi.values() if any(p.proposed_name for p in mem))
            conflicted = sum(1 for mem in multi.values() if any(p.status == "conflicted" for p in mem))
            anchor_less = sum(1 for mem in multi.values() if all(p.status == "ask_name" for p in mem))
            singletons = sum(1 for mem in clusters_by_id.values() if len(mem) == 1)
            proposals_count = sum(1 for p in props if p.status == "proposed")
            ask_for_name = sum(1 for p in props if p.status == "ask_name")

            def pct(n: int) -> float:
                return 100.0 * n / clusters if clusters else 0.0

            stats.append(StatsPane(
                window=window,
                appearances=len(props),
                clusters=clusters,
                anchored_pct=pct(anchored),
                conflicted_pct=pct(conflicted),
                anchor_less_pct=pct(anchor_less),
                singletons=singletons,
                proposals=proposals_count,
                ask_for_name=ask_for_name,
            ))
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
        from datetime import datetime, timezone

        cluster = int(body.get("cluster", -1))
        window = str(body.get("window", ""))
        name = str(body.get("name", "")).strip()
        included = body.get("included", [])
        dry_run = bool(body.get("dry_run", True))

        if cluster < 0 or not window or not name:
            return {"error": "missing required fields"}

        if not included:
            return {"error": "no appearances included"}

        # Get the proposals for this cluster
        proposals = self.store.proposals()
        target_props = [
            p for p in proposals
            if p.cluster == cluster and p.window == window and p.status == "proposed"
        ]

        if not target_props:
            return {"error": "no writable proposals found for this cluster"}

        # Build a set of included (asset_id, appearance) for quick lookup
        included_set = {
            (item["asset_id"], item["appearance"])
            for item in included
            if "asset_id" in item and "appearance" in item
        }

        # Filter to only included proposals
        write_props = [p for p in target_props if (p.asset_id, p.appearance) in included_set]

        if not write_props:
            return {"error": "none of the included appearances are writable"}

        # Resolve the target person
        if dry_run:
            # In dry-run, we don't actually create the person or write faces
            # Just return what would happen
            person_id = "dry-run-person-id"
            results = []
            for prop in write_props:
                appearances_by_asset = self.store.appearances_by_asset()
                app_list = appearances_by_asset.get(prop.asset_id, [])
                appearance = next((a for a in app_list if a.appearance == prop.appearance), None)
                if appearance is None:
                    results.append({
                        "asset_id": prop.asset_id,
                        "appearance": prop.appearance,
                        "status": "error",
                        "message": "appearance not found",
                    })
                    continue

                # Check if already written (via write ledger)
                if self.store.has_write_ledger_entry(prop.asset_id, person_id):
                    results.append({
                        "asset_id": prop.asset_id,
                        "appearance": prop.appearance,
                        "status": "skipped",
                        "message": "already written (ledger)",
                    })
                    continue

                # Check tombstone
                if self.store.has_tombstone(prop.asset_id, person_id):
                    results.append({
                        "asset_id": prop.asset_id,
                        "appearance": prop.appearance,
                        "status": "skipped",
                        "message": "tombstone (manually removed in Immich)",
                    })
                    continue

                # Resolve head box
                person_box = appearance.box
                head_box = self.immich.resolve_head_box(
                    prop.asset_id,
                    person_box,
                    self.cfg.clustering.face_overlap,
                    pixel_source=self.cfg.detection.pixel_source,
                )
                if head_box is None:
                    results.append({
                        "asset_id": prop.asset_id,
                        "appearance": prop.appearance,
                        "status": "skipped",
                        "message": "no head found",
                    })
                    continue

                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "would_write",
                    "head_box": head_box,
                })

            return {
                "cluster": cluster,
                "window": window,
                "name": name,
                "dry_run": True,
                "results": results,
            }

        # Real write path: need write-scoped API key
        write_api_key = self.cfg.immich.write_api_key
        if not write_api_key:
            return {"error": "write-scoped API key not configured"}

        # Create a write-scoped client (with head detector for fallback)
        head_cfg = HeadDetectorConfig(
            weights_path=self.cfg.head_detection.weights_path,
            input_size=self.cfg.head_detection.input_size,
            confidence_threshold=self.cfg.head_detection.confidence_threshold,
            top_expansion_ratio=self.cfg.head_detection.top_expansion_ratio,
            width_expansion_ratio=self.cfg.head_detection.width_expansion_ratio,
            nms_threshold=self.cfg.head_detection.nms_threshold,
        )
        write_client = ImmichClient(
            self.cfg.immich.base_url,
            write_api_key,
            page_size=self.cfg.immich.page_size,
            timeout=self.cfg.immich.timeout_seconds,
            head_detector_cfg=head_cfg,
        )

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
        timestamp = datetime.now(timezone.utc).isoformat()

        for prop in write_props:
            appearances_by_asset = self.store.appearances_by_asset()
            app_list = appearances_by_asset.get(prop.asset_id, [])
            appearance = next((a for a in app_list if a.appearance == prop.appearance), None)
            if appearance is None:
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "error",
                    "message": "appearance not found",
                })
                continue

            # Check if already written (via write ledger) - idempotency
            if self.store.has_write_ledger_entry(prop.asset_id, person_id):
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "skipped",
                    "message": "already written (ledger)",
                })
                continue

            # Check tombstone
            if self.store.has_tombstone(prop.asset_id, person_id):
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "skipped",
                    "message": "tombstone (manually removed in Immich)",
                })
                continue

            # Check existing faces for conflicts and already-labeled
            person_box = appearance.box
            has_conflict, already_labeled = write_client.check_existing_faces(
                prop.asset_id, person_box, person_id, self.cfg.clustering.face_overlap
            )

            if already_labeled:
                # Already has this person's face - skip and journal
                self.store.add_write_ledger_entry(WriteLedgerEntry(
                    asset_id=prop.asset_id,
                    person_id=person_id,
                    cluster=cluster,
                    window=window,
                    timestamp=timestamp,
                ))
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "skipped",
                    "message": "already labeled in Immich",
                })
                continue

            if has_conflict:
                # Different person's overlapping face - conflict
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "skipped",
                    "message": "conflict: different person's overlapping face",
                })
                continue

            # Resolve head box
            head_box = write_client.resolve_head_box(prop.asset_id, person_box, self.cfg.clustering.face_overlap)
            if head_box is None:
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "skipped",
                    "message": "no head found",
                })
                continue

            # Write the face
            try:
                write_client.create_face(
                    prop.asset_id,
                    person_id,
                    head_box[0],
                    head_box[1],
                    head_box[2],
                    head_box[3],
                )
            except ImmichError as exc:
                results.append({
                    "asset_id": prop.asset_id,
                    "appearance": prop.appearance,
                    "status": "error",
                    "message": f"face creation failed: {exc.message}",
                })
                continue

            # Journal the write
            self.store.add_write_ledger_entry(WriteLedgerEntry(
                asset_id=prop.asset_id,
                person_id=person_id,
                cluster=cluster,
                window=window,
                timestamp=timestamp,
            ))

            results.append({
                "asset_id": prop.asset_id,
                "appearance": prop.appearance,
                "status": "written",
                "head_box": head_box,
            })

        return {
            "cluster": cluster,
            "window": window,
            "name": name,
            "person_id": person_id,
            "dry_run": False,
            "results": results,
        }


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
    @click.option("--config", "config_path", required=True, type=click.Path(path_type=Path, dir_okay=False))
    @click.option("--host", default="0.0.0.0", help="Bind address")
    @click.option("--port", default=8080, type=int, help="Port to listen on")
    def ui(config_path: Path, host: str, port: int) -> None:
        """Run the confirmation UI HTTP server."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
        serve(config_path, host, port)

    ui()


if __name__ == "__main__":
    main()