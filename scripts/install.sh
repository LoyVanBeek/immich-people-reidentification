#!/usr/bin/env bash
# scripts/install.sh — Idempotent, mode-aware installer for Immich Person Re-ID
#
# Modes:
#   auto    — compose if reid-core/reid-ui containers exist, else systemd
#   compose — run via docker compose (local image reid-batch:local or ghcr)
#   systemd — run via systemd units (ghcr image)
#
# Idempotent: re-running converges whatever is currently installed (compose or
# systemd), never breaks it. A second run with no changes is a no-op.
#
# Run as the pi user (not root) on the Raspberry Pi 5.
# Usage: ./scripts/install.sh [--mode auto|compose|systemd] [--install-dir DIR]
#        [--data-dir DIR] [--user USER] [--image IMAGE] [--up] [--dry-run]

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Globals (set by parse_args, then defaults applied in main)
# ---------------------------------------------------------------------------
MODE="auto"
INSTALL_DIR=""
DATA_DIR=""
SERVICE_USER=""
IMAGE=""
UP="false"
DRY_RUN="false"
# Services (core/ui) that were running before the install began — they are
# stopped before the build and started again after it. Compose service names
# or systemd unit names, depending on MODE.
PREV_ACTIVE=""
# Compose services to `up -d` after install (empty = nothing was running).
COMPOSE_TARGETS=""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --mode MODE           auto | compose | systemd  (default: auto)
                        auto = compose if reid-core/reid-ui containers exist,
                        else systemd
  --install-dir DIR     Where the repo lives (default: directory the script
                        lives in's parent = repo root)
  --data-dir DIR        Data directory for the SQLite store
                        (default: REID_DATA_DIR from .env, or <install-dir>/reid-data)
  --user USER           System user to run services as (default: current user)
  --image IMAGE         Container image (default: auto-detect reid-batch:local,
                        else ghcr.io/loy/immich-person-reidentification:latest)
  --up                  Start services after installing
  --dry-run             Show what would be done without making changes
  -h, --help            Show this help

After a git pull, re-run this script: if the local image reid-batch:local is in
use, it is rebuilt when the git HEAD changed since the last build (marker:
.reid-image-commit), so the new code actually runs. The ghcr :latest image is
pulled by the units on every start.

This script:
  * Writes/merges reid.toml (generates a core API key, sets core host to
    0.0.0.0 so a remote UI can reach it) — never overwrites an existing file.
  * Creates the data directory.
  * Compose mode: stops/disables any colliding systemd core/UI units, then
    brings up docker compose. Never enables systemd units in compose mode.
  * Systemd mode: renders unit files, installs only changed ones, stops any
    compose containers, then daemon-reload/enable/start only changed units.
  * Second run with no changes is a no-op.
  * Stops the running reid-core/reid-ui before building/installing and
    starts them again after (only those that were running, unless --up),
    so a rebuilt image always replaces the old containers.

Prerequisites:
  - Docker installed and user in the docker group
  - Two Immich API keys created (read-scoped and write-scoped) — set them in
    reid.toml after the first run.

EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                MODE="$2"; shift 2 ;;
            --install-dir)
                INSTALL_DIR="$2"; shift 2 ;;
            --data-dir)
                DATA_DIR="$2"; shift 2 ;;
            --user)
                SERVICE_USER="$2"; shift 2 ;;
            --image)
                IMAGE="$2"; shift 2 ;;
            --up)
                UP="true"; shift ;;
            --dry-run)
                DRY_RUN="true"; shift ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
apply_defaults() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    INSTALL_DIR="${INSTALL_DIR:-${REPO_ROOT}}"

    # Read REID_DATA_DIR from .env if present (never overwrite an existing .env).
    # Source it safely: only pick up REID_DATA_DIR, never UID/GID which are
    # readonly shell variables and would abort the script under `set -u`.
    if [[ -f "${INSTALL_DIR}/.env" ]]; then
        REID_DATA_DIR="$(grep -E '^REID_DATA_DIR=' "${INSTALL_DIR}/.env" \
            | head -n1 | cut -d= -f2-)" || true
    fi
    DATA_DIR="${DATA_DIR:-${REID_DATA_DIR:-${INSTALL_DIR}/reid-data}}"

    SERVICE_USER="${SERVICE_USER:-${USER}}"
    HOME_DIR="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
    HOME_DIR="${HOME_DIR:-/home/${SERVICE_USER}}"
    UID_GID="$(id -u "${SERVICE_USER}"):$(id -g "${SERVICE_USER}")"

    GHCR_IMAGE="ghcr.io/loy/immich-person-reidentification:latest"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
have_docker() {
    command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

have_systemd() {
    command -v systemctl >/dev/null 2>&1 && systemctl --version >/dev/null 2>&1
}

# True when reid-core/reid-ui compose containers exist (created, running or
# exited) — used by auto mode and by systemd mode to know what to stop.
is_compose_running() {
    have_docker || return 1
    local n
    n="$(docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
        ps --filter "name=reid-core" --filter "name=reid-ui" -q 2>/dev/null | wc -l)"
    [[ "${n}" -gt 0 ]]
}

# True when a systemd unit is enabled (is-enabled returns 0).
unit_enabled() {
    have_systemd || return 1
    systemctl is-enabled "$1" >/dev/null 2>&1
}

# True when a systemd unit is active.
unit_active() {
    have_systemd || return 1
    systemctl is-active "$1" >/dev/null 2>&1
}

# Detect image: reid-batch:local if present, else ghcr.
detect_image() {
    if [[ -n "${IMAGE}" ]]; then
        echo "${IMAGE}"
        return
    fi
    if have_docker && docker image inspect reid-batch:local >/dev/null 2>&1; then
        echo "reid-batch:local"
    else
        echo "${GHCR_IMAGE}"
    fi
}

# Which long-running services (reid-core, reid-ui) are currently active, as
# a space-separated list of compose service names or systemd unit names.
# Empty in dry-run (no state probing) and when nothing is running.
active_services() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo ""
        return 0
    fi
    if [[ "${MODE}" == "compose" ]]; then
        have_docker || { echo ""; return 0; }
        docker compose -f "${INSTALL_DIR}/docker-compose.yml" --profile local \
            ps --format '{{.Name}}' 2>/dev/null \
            | grep -E '^(reid-core|reid-ui)$' | sort -u | tr '\n' ' ' | sed 's/ $//'
    else
        have_systemd || { echo ""; return 0; }
        local out=()
        local u
        for u in reid-core.service reid-ui.service; do
            if systemctl is-active "${u}" >/dev/null 2>&1; then
                out+=("${u}")
            fi
        done
        echo "${out[*]}"
    fi
}

# Stop the running services we are about to reinstall, so a rebuilt image can
# never leave two containers bound to the same ports. They are started again
# by start_services / install_compose after the install.
stop_services() {
    local active="$1"
    if [[ -z "${active}" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would stop any running reid-core/reid-ui before installing"
        else
            log_info "No reid core/ui services currently running"
        fi
        return 0
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would stop: ${active}"
        return 0
    fi
    if [[ "${MODE}" == "compose" ]]; then
        log_info "Stopping compose services: ${active}"
        docker compose -f "${INSTALL_DIR}/docker-compose.yml" --profile local stop ${active}
    else
        log_info "Stopping systemd units: ${active}"
        sudo systemctl stop ${active}
    fi
}

# After installing (systemd mode; compose starts its targets in
# install_compose): start the services that were running before, or all of
# them with --up.
start_services() {
    local previously="$1"
    local targets="${previously}"
    if [[ "${UP}" == "true" ]]; then
        targets="reid-core.service reid-ui.service"
    fi
    if [[ -z "${targets}" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would restart previously running services after install (none detected in dry-run)"
        else
            log_info "Nothing was running before install — not starting (use --up to start)"
        fi
        return 0
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would start after install: ${targets}"
        return 0
    fi
    log_info "Starting systemd units: ${targets}"
    local u
    for u in ${targets}; do
        sudo systemctl enable "${u}" >/dev/null 2>&1 || true
        sudo systemctl start "${u}"
    done
}

# Rebuild reid-batch:local when the repo's git HEAD changed since the last
# build (or the image is missing) — so `git pull && ./scripts/install.sh`
# always deploys the new code, not the stale local image. The ghcr image is
# unaffected: the units pull :latest on start.
ensure_local_image() {
    [[ "${IMAGE}" == "reid-batch:local" ]] || return 0
    local head marker="${INSTALL_DIR}/.reid-image-commit" built=""
    head="$(git -C "${INSTALL_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [[ -f "${marker}" ]]; then
        built="$(cat "${marker}" 2>/dev/null || true)"
    fi
    if have_docker && docker image inspect reid-batch:local >/dev/null 2>&1 \
        && [[ "${built}" == "${head}" ]]; then
        log_info "Local image reid-batch:local up to date (${head})"
        return 0
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would build reid-batch:local at ${head} (build-arg GIT_COMMIT=${head})"
        return 0
    fi
    if ! have_docker; then
        log_warn "Docker unavailable; cannot rebuild reid-batch:local (using whatever is present)"
        return 0
    fi
    log_info "Building reid-batch:local at ${head} (first build is slow, later ones are cached)..."
    (cd "${INSTALL_DIR}" && docker buildx build \
        --platform linux/arm64 -t reid-batch:local \
        --build-arg GIT_COMMIT="${head}" .)
    echo "${head}" > "${marker}"
    log_info "Built reid-batch:local (${head})"
}

# Generate a random core API key (hex) if openssl is available, else /dev/urandom.
generate_api_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}

# ---------------------------------------------------------------------------
# reid.toml — create if missing, merge [core] settings if present
# ---------------------------------------------------------------------------
ensure_reid_toml() {
    local config="${INSTALL_DIR}/reid.toml"

    if [[ ! -f "${config}" ]]; then
        local key
        key="$(generate_api_key)"
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would create ${config} (core host=0.0.0.0, generated api_key)"
        else
            log_info "Creating ${config} (core host=0.0.0.0, generated api_key)"
            cat > "${config}" <<TOMLEOF
[immich]
# Immich API base URL (include port if not 80/443)
base_url = "http://localhost:2283"

# Read-scoped API key — used for everything except writes. Create in Immich
# under Account Settings → API Keys with exactly:
#   asset.read  asset.view  asset.download  face.read  person.read
# (or export IMMICH_API_KEY instead of putting it in this file)
api_key = "CHANGE_ME_READ_KEY"

# Write-scoped API key — used ONLY by "Write … to Immich" in the Final check
# tab. Create with exactly:
#   person.read  person.create  face.read  face.create
# Leave unset during development and the UI cannot write to Immich at all.
# (or export IMMICH_WRITE_API_KEY)
write_api_key = "CHANGE_ME_WRITE_KEY"

page_size = 1000
timeout_seconds = 30

[store]
# SQLite database path (mounted at /data inside container)
path = "/data/reid.sqlite3"

[run]
log_level = "info"

[core]
# The core API server (reid-core) owns the store; the UI proxies /api/*
# to it. Bound to 0.0.0.0 so a UI on another machine can reach it;
# protected by api_key below.
url = "http://127.0.0.1:8090"
host = "0.0.0.0"
port = 8090
# Generated core API key — the UI and nightly trigger present this.
api_key = "${key}"

[ui]
# Override for the Immich URL used by "open in Immich" links when the UI
# runs on another machine. The core reports its own base_url, which on the
# Pi is usually localhost and means nothing on a laptop.
# immich_url = "http://<pi>:2283"

[detection]
confidence_floor = 0.35
min_box_height = 100
pixel_source = "preview"

[clustering]
window_days = 1
distance_threshold = 0.5
min_cluster_size = 2
face_overlap = 0.5

[head_detection]
input_size = 320
confidence_threshold = 0.5
top_expansion_ratio = 0.55
width_expansion_ratio = 1.8
nms_threshold = 0.4
TOMLEOF
        fi
        return
    fi

    # File exists: merge [core] settings without touching anything else.
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would ensure [core] host/api_key in ${config}"
        return
    fi

    local key
    key="$(generate_api_key)"
    python3 - "${config}" "${key}" <<'PYEOF'
import sys

config_path = sys.argv[1]
new_key = sys.argv[2]

with open(config_path) as fh:
    lines = fh.read().split("\n")

# Locate [core] section (if any)
core_idx = None
for i, line in enumerate(lines):
    if line.strip() == "[core]":
        core_idx = i
        break

if core_idx is None:
    # Append a [core] section at the end
    lines.append("")
    lines.append("[core]")
    lines.append('url = "http://127.0.0.1:8090"')
    lines.append('host = "0.0.0.0"')
    lines.append("port = 8090")
    lines.append(f'api_key = "{new_key}"')
else:
    # Find the end of the [core] section (next [section] or EOF)
    section_end = len(lines)
    for i in range(core_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("[") and stripped.endswith("]") and stripped != "[core]":
            section_end = i
            break

    # Scan existing keys within [core]
    has_host = False
    has_api_key = False
    for i in range(core_idx + 1, section_end):
        stripped = lines[i].strip()
        if stripped.startswith("host"):
            has_host = True
        elif stripped.startswith("api_key"):
            has_api_key = True

    insert_at = section_end
    if not has_host:
        lines.insert(insert_at, 'host = "0.0.0.0"')
        insert_at += 1
    if not has_api_key:
        lines.insert(insert_at, f'api_key = "{new_key}"')

with open(config_path, "w") as fh:
    fh.write("\n".join(lines))
PYEOF
    log_info "Ensured [core] host=0.0.0.0 and api_key in ${config}"
    log_warn "Edit ${config} to add your Immich read/write API keys!"
}

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
ensure_data_dir() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would create data directory: ${DATA_DIR}"
        return
    fi
    mkdir -p "${DATA_DIR}"
    log_info "Data directory: ${DATA_DIR}"
}

# ---------------------------------------------------------------------------
# Unit file rendering
# ---------------------------------------------------------------------------
# Renders a unit template (stdin) with variable substitution to stdout.
render_unit() {
    local install_dir="$1" user="$2" group="$3" uid_gid="$4" image="$5" data_dir="$6"
    sed \
        -e "s|@@INSTALL_DIR@@|${install_dir}|g" \
        -e "s|@@USER@@|${user}|g" \
        -e "s|@@GROUP@@|${group}|g" \
        -e "s|@@UID_GID@@|${uid_gid}|g" \
        -e "s|@@IMAGE@@|${image}|g" \
        -e "s|@@DATA_DIR@@|${data_dir}|g"
}

render_reid_core() {
    local install_dir="$1" user="$2" group="$3" uid_gid="$4" image="$5" data_dir="$6"
    cat <<'UNITEOF' | render_unit "${install_dir}" "${user}" "${group}" "${uid_gid}" "${image}" "${data_dir}"
# reid-core.service — Core API server for Immich Person Re-ID
# Installed by scripts/install.sh to /etc/systemd/system/reid-core.service
#
# This is a long-running service that owns the store and serves the core
# API on port 8090 (by default); reid-ui proxies to it. It pulls the image
# from ghcr.io.
#
# To enable: systemctl enable --now reid-core.service
# To view logs: journalctl -u reid-core -f

[Unit]
Description=Immich Person Re-ID core API server (store owner)
Documentation=https://github.com/loy/immich-person-reidentification
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=@@USER@@
Group=@@GROUP@@
WorkingDirectory=@@INSTALL_DIR@@
Environment=GHCR_IMAGE=@@IMAGE@@
# Pull latest image on start; ignore pull failures (offline Pi, etc.)
ExecStartPre=-/usr/bin/docker pull ${GHCR_IMAGE}
# Stop and remove any previously running container with the same name so a
# restart never leaves two containers bound to port 8090.
ExecStartPre=-/usr/bin/docker stop reid-core
ExecStartPre=-/usr/bin/docker rm -f reid-core
# Run the core API server — override ENTRYPOINT (image default is reid-batch)
ExecStart=/usr/bin/docker run --rm \
  --platform linux/arm64 \
  --network host \
  --name reid-core \
  --user @@UID_GID@@ \
  --entrypoint reid-core \
  -v @@INSTALL_DIR@@/reid.toml:/etc/reid/reid.toml:ro \
  -v @@DATA_DIR@@:/data \
  ${GHCR_IMAGE} \
  --config /etc/reid/reid.toml
Restart=on-failure
RestartSec=30
TimeoutStopSec=15
# Stop the named container gracefully BEFORE systemd signals the docker run
# client; without this the container keeps running after the client dies.
ExecStop=/usr/bin/docker stop -t 10 reid-core
# Force-remove any leftover container (running or stopped).
ExecStopPost=-/usr/bin/docker rm -f reid-core
StandardOutput=journal
StandardError=journal
SyslogIdentifier=reid-core

[Install]
WantedBy=multi-user.target
UNITEOF
}

render_reid_ui() {
    local install_dir="$1" user="$2" group="$3" uid_gid="$4" image="$5" data_dir="$6"
    cat <<'UNITEOF' | render_unit "${install_dir}" "${user}" "${group}" "${uid_gid}" "${image}" "${data_dir}"
# reid-ui.service — Confirmation UI for Immich Person Re-ID
# Installed by scripts/install.sh to /etc/systemd/system/reid-ui.service
#
# This is a long-running service that serves the confirmation UI on
# port 8080 (by default). It proxies /api/* to reid-core. It pulls the image
# from ghcr.io.
#
# To enable: systemctl enable --now reid-ui.service
# To view logs: journalctl -u reid-ui -f

[Unit]
Description=Immich Person Re-ID confirmation UI
Documentation=https://github.com/loy/immich-person-reidentification
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=@@USER@@
Group=@@GROUP@@
WorkingDirectory=@@INSTALL_DIR@@
Environment=GHCR_IMAGE=@@IMAGE@@
# Pull latest image on start; ignore pull failures
ExecStartPre=-/usr/bin/docker pull ${GHCR_IMAGE}
# Stop and remove any previously running container with the same name so a
# restart never leaves two containers bound to port 8080.
ExecStartPre=-/usr/bin/docker stop reid-ui
ExecStartPre=-/usr/bin/docker rm -f reid-ui
# Run the UI server — override ENTRYPOINT (image default is reid-batch).
# No store volume: the UI proxies /api/* to reid-core and never touches the
# database itself.
ExecStart=/usr/bin/docker run --rm \
  --platform linux/arm64 \
  --network host \
  --name reid-ui \
  --user @@UID_GID@@ \
  --entrypoint reid-ui \
  -v @@INSTALL_DIR@@/reid.toml:/etc/reid/reid.toml:ro \
  ${GHCR_IMAGE} \
  --config /etc/reid/reid.toml
Restart=on-failure
RestartSec=30
TimeoutStopSec=15
# Stop the named container gracefully BEFORE systemd signals the docker run
# client; without this the container keeps running after the client dies.
ExecStop=/usr/bin/docker stop -t 10 reid-ui
# Force-remove any leftover container (running or stopped).
ExecStopPost=-/usr/bin/docker rm -f reid-ui
StandardOutput=journal
StandardError=journal
SyslogIdentifier=reid-ui

[Install]
WantedBy=multi-user.target
UNITEOF
}

render_reid_batch_service() {
    local install_dir="$1" user="$2" group="$3" uid_gid="$4" image="$5" data_dir="$6"
    cat <<'UNITEOF' | render_unit "${install_dir}" "${user}" "${group}" "${uid_gid}" "${image}" "${data_dir}"
# reid-batch.service — Nightly pipeline trigger for Immich Person Re-ID
# Installed by scripts/install.sh to /etc/systemd/system/reid-batch.service
#
# This service asks the core (reid-core) to start one pipeline run, then
# exits. The core owns the store and runs the pipeline as a subprocess, so
# this unit never touches the database and never runs two runs at once.
# It is triggered by reid-batch.timer (see below) on a nightly schedule.
#
# The container image is pulled from ghcr.io. If you build locally, change
# the image in the installed unit.

[Unit]
Description=Immich Person Re-ID nightly run trigger
Documentation=https://github.com/loy/immich-person-reidentification
After=network-online.target docker.service reid-core.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
User=@@USER@@
Group=@@GROUP@@
WorkingDirectory=@@INSTALL_DIR@@
Environment=GHCR_IMAGE=@@IMAGE@@
# Pull latest image before each run; ignore pull failures
ExecStartPre=-/usr/bin/docker pull ${GHCR_IMAGE}
# Trigger one run on the core and wait for it to finish. No store volume:
# the core owns the store.
ExecStart=/usr/bin/docker run --rm \
  --platform linux/arm64 \
  --network host \
  --user @@UID_GID@@ \
  --entrypoint reid-trigger \
  -v @@INSTALL_DIR@@/reid.toml:/etc/reid/reid.toml:ro \
  ${GHCR_IMAGE} \
  --config /etc/reid/reid.toml --trigger timer --wait
TimeoutSec=21600
StandardOutput=journal
StandardError=journal
SyslogIdentifier=reid-batch

[Install]
WantedBy=multi-user.target
UNITEOF
}

render_reid_batch_timer() {
    cat <<UNITEOF
# reid-batch.timer — Nightly trigger for Immich Person Re-ID batch
# Installed by scripts/install.sh to /etc/systemd/system/reid-batch.timer
#
# Runs daily at 02:00 (adjust OnCalendar as needed). reid-batch.service
# is a oneshot trigger: it asks the core for a run and waits for it.
#
# To enable: systemctl enable --now reid-batch.timer
# To check:  systemctl list-timers reid-batch.timer
# To run now: systemctl start reid-batch.service

[Unit]
Description=Nightly Immich Person Re-ID run trigger
Requires=reid-batch.service

[Timer]
# Run daily at 02:00 local time. Change as needed:
#   OnCalendar=*-*-* 02:00:00
#   OnCalendar=*-*-* 03:30:00
# See systemd.time(7) for format.
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=15m
AccuracySec=1m

[Install]
WantedBy=timers.target
UNITEOF
}

# Install a unit file only if its rendered content differs from the installed
# version. Returns 0 if changed/installed, 1 if unchanged.
install_unit() {
    local name="$1" rendered="$2"
    local dst="/etc/systemd/system/${name}"

    if [[ -f "${dst}" ]] && cmp -s "${rendered}" "${dst}"; then
        log_info "${name}: unchanged"
        return 1
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would install ${name} -> ${dst}"
    else
        sudo cp "${rendered}" "${dst}"
        log_info "Installed ${name}"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Systemd mode
# ---------------------------------------------------------------------------
install_systemd() {
    if [[ "${DRY_RUN}" != "true" ]] && ! have_systemd; then
        log_error "systemd is not available on this host"
        exit 1
    fi

    local image
    image="$(detect_image)"

    local changed=()

    # Render units to a temp dir
    tmpdir="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '${tmpdir}'" RETURN

    render_reid_core "${INSTALL_DIR}" "${SERVICE_USER}" "${SERVICE_USER}" "${UID_GID}" "${image}" "${DATA_DIR}" > "${tmpdir}/reid-core.service"
    render_reid_ui "${INSTALL_DIR}" "${SERVICE_USER}" "${SERVICE_USER}" "${UID_GID}" "${image}" "${DATA_DIR}" > "${tmpdir}/reid-ui.service"
    render_reid_batch_service "${INSTALL_DIR}" "${SERVICE_USER}" "${SERVICE_USER}" "${UID_GID}" "${image}" "${DATA_DIR}" > "${tmpdir}/reid-batch.service"
    render_reid_batch_timer > "${tmpdir}/reid-batch.timer"

    for unit in reid-core.service reid-ui.service reid-batch.service reid-batch.timer; do
        if install_unit "${unit}" "${tmpdir}/${unit}"; then
            changed+=("${unit}")
        fi
    done

    # Stop compose containers if any (mode switch: compose -> systemd)
    if [[ "${DRY_RUN}" != "true" ]] && is_compose_running; then
        log_info "Stopping compose containers (switching to systemd mode)..."
        docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
            --profile local down 2>/dev/null || true
        docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
            --profile remote down 2>/dev/null || true
    fi

    if [[ "${#changed[@]}" -eq 0 ]]; then
        log_info "No systemd units changed."
    else
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would run: systemctl daemon-reload"
            for unit in "${changed[@]}"; do
                log_info "Would enable/start: ${unit}"
            done
        else
            log_info "Reloading systemd..."
            sudo systemctl daemon-reload

            for unit in "${changed[@]}"; do
                case "${unit}" in
                    reid-batch.timer)
                        if [[ "${UP}" == "true" ]]; then
                            sudo systemctl enable --now "${unit}"
                        else
                            sudo systemctl enable "${unit}"
                        fi
                        ;;
                    reid-core.service|reid-ui.service)
                        if [[ "${UP}" == "true" ]]; then
                            sudo systemctl enable --now "${unit}"
                        else
                            sudo systemctl enable "${unit}"
                        fi
                        ;;
                    reid-batch.service)
                        sudo systemctl enable "${unit}"
                        ;;
                esac
            done
        fi
    fi

    if [[ "${DRY_RUN}" != "true" ]] && [[ "${#changed[@]}" -gt 0 ]] && [[ "${UP}" == "true" ]]; then
        log_info "Waiting for services to come up..."
        sleep 3
    fi
}

# ---------------------------------------------------------------------------
# Compose mode
# ---------------------------------------------------------------------------
install_compose() {
    if [[ "${DRY_RUN}" != "true" ]] && ! have_docker; then
        log_error "Docker is not available on this host"
        exit 1
    fi

    local image
    image="$(detect_image)"

    # Stop/disable any colliding systemd core/UI units (mode switch: systemd -> compose)
    if [[ "${DRY_RUN}" != "true" ]] && have_systemd; then
        for unit in reid-core.service reid-ui.service; do
            if unit_enabled "${unit}"; then
                log_info "Stopping/disabling ${unit} (compose mode)..."
                sudo systemctl disable --now "${unit}" 2>/dev/null || true
            fi
        done
    fi

    # Write .env for compose (UID/GID, data dir)
    local env_file="${INSTALL_DIR}/.env"
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would write ${env_file} (UID_GID=${UID_GID}, REID_DATA_DIR=${DATA_DIR})"
    else
        {
            echo "UID=$(id -u "${SERVICE_USER}")"
            echo "GID=$(id -g "${SERVICE_USER}")"
            echo "REID_DATA_DIR=${DATA_DIR}"
        } > "${env_file}"
        log_info "Wrote ${env_file}"
    fi

    local targets="${COMPOSE_TARGETS:-reid-core reid-ui}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would run: docker compose --profile local up -d ${targets}"
        if [[ "${UP}" == "true" ]]; then
            log_info "Would start services"
        fi
    elif [[ -z "${targets}" ]]; then
        log_info "No compose services were running before install; not starting (use --up to start)"
    else
        log_info "Bringing up docker compose (profile: local): ${targets}..."
        # --build rebuilds reid-batch:local when sources changed (cached, so
        # a no-change run is fast); without it compose reuses the old image.
        local build_flag=""
        if [[ "${IMAGE}" == "reid-batch:local" ]]; then
            build_flag="--build"
        fi
        docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
            --profile local up -d ${build_flag} ${targets}

        if [[ "${UP}" == "true" ]]; then
            log_info "Waiting for core and UI to be ready..."
            local ready=0
            for i in $(seq 1 30); do
                if docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
                    --profile local ps reid-core reid-ui 2>/dev/null | \
                    grep -q "Up"; then
                    ready=1
                    break
                fi
                sleep 1
            done
            if [[ "${ready}" -eq 0 ]]; then
                log_warn "Containers did not report 'Up' within 30s; check docker compose logs."
            fi
        fi
    fi
}

# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
health_check() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return
    fi

    log_info "Health check:"
    if [[ "${MODE}" == "compose" ]]; then
        if have_docker; then
            docker compose -f "${INSTALL_DIR}/docker-compose.yml" \
                --profile local ps reid-core reid-ui 2>/dev/null || true
        fi
    else
        if have_systemd; then
            systemctl is-active reid-core.service 2>/dev/null && \
                log_info "reid-core.service: active" || log_warn "reid-core.service: not active"
            systemctl is-active reid-ui.service 2>/dev/null && \
                log_info "reid-ui.service: active" || log_warn "reid-ui.service: not active"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    apply_defaults

    log_info "Mode: ${MODE}"
    log_info "Install dir: ${INSTALL_DIR}"
    log_info "Data dir: ${DATA_DIR}"
    log_info "User: ${SERVICE_USER} (${UID_GID})"

    # Validate repo
    if [[ ! -f "${INSTALL_DIR}/docker-compose.yml" ]]; then
        log_error "Not a reid repo at ${INSTALL_DIR} (docker-compose.yml missing)"
        exit 1
    fi

    # Auto-detect mode
    if [[ "${MODE}" == "auto" ]]; then
        if is_compose_running; then
            MODE="compose"
        else
            MODE="systemd"
        fi
        log_info "Auto-detected mode: ${MODE}"
    fi

    # Detect image (may need docker; skip in dry-run)
    if [[ "${DRY_RUN}" != "true" ]]; then
        IMAGE="$(detect_image)"
    else
        IMAGE="${IMAGE:-${GHCR_IMAGE}}"
    fi
    log_info "Image: ${IMAGE}"

    # Stop the services we are about to reinstall BEFORE building/installing,
    # so a rebuilt image can never leave two containers bound to the same
    # ports. They are started again after the install — only those that were
    # running before, unless --up.
    PREV_ACTIVE="$(active_services)"
    stop_services "${PREV_ACTIVE}"
    if [[ "${MODE}" == "compose" ]]; then
        if [[ "${UP}" == "true" ]]; then
            COMPOSE_TARGETS="reid-core reid-ui"
        else
            COMPOSE_TARGETS="${PREV_ACTIVE}"
        fi
    fi

    # Rebuild the local image when the git HEAD moved since the last build,
    # so a git pull + re-run of this script deploys the new code.
    export GIT_COMMIT="$(git -C "${INSTALL_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    ensure_local_image

    # reid.toml + data dir
    ensure_reid_toml
    ensure_data_dir

    # Mode-specific install; then restart what was running before.
    if [[ "${MODE}" == "compose" ]]; then
        install_compose
    else
        install_systemd
        start_services "${PREV_ACTIVE}"
    fi

    health_check

    log_info ""
    log_info "=== Installation complete ==="
    log_info ""
    log_info "  Config:    ${INSTALL_DIR}/reid.toml  (add your Immich API keys!)"
    log_info "  Data dir:  ${DATA_DIR}"
    if [[ "${MODE}" == "compose" ]]; then
        log_info "  UI:        http://<pi>:8080   (docker compose --profile local)"
        log_info "  Core:      http://<pi>:8090   (bound 0.0.0.0, API key protected)"
    else
        log_info "  systemctl status reid-core reid-ui"
        log_info "  journalctl -u reid-core -f"
        log_info "  journalctl -u reid-ui -f"
    fi
}

main "$@"
