#!/usr/bin/env bash
# install-systemd.sh — Install systemd services for Immich Person Re-ID
#
# Run as the pi user (not root) on the Raspberry Pi 5.
# Usage: ./scripts/install-systemd.sh [--user pi] [--image ghcr.io/owner/repo:tag]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_USER="${SUDO_USER:-${USER}}"
GHCR_IMAGE="ghcr.io/loy/immich-person-reidentification:latest"
INSTALL_DIR="/home/${SERVICE_USER}/immich-person-reidentification"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --user USER          System user to run services as (default: current user)
  --image IMAGE        ghcr.io image to pull (default: ${GHCR_IMAGE})
  --install-dir DIR    Where the repo lives on the Pi (default: /home/USER/immich-person-reidentification)
  --dry-run            Show what would be done without making changes
  -h, --help           Show this help

This script:
  1. Creates the reid.toml config from template if missing
  2. Creates reid-data directory
  3. Installs systemd service files to /etc/systemd/system/
  4. Reloads systemd and enables the timer + UI service

Prerequisites:
  - Docker installed and user in docker group
  - ghcr.io image exists (pushed via CI on version tags) OR build locally first
  - Two Immich API keys created (read-scoped and write-scoped)

EOF
}

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Parse args
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        --image)
            GHCR_IMAGE="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

HOME_DIR="/home/${SERVICE_USER}"
INSTALL_DIR="${INSTALL_DIR:-${HOME_DIR}/immich-person-reidentification}"

if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "DRY RUN — would execute the following:"
fi

# 1. Ensure repo directory exists
if [[ ! -d "${INSTALL_DIR}" ]]; then
    log_error "Repository not found at ${INSTALL_DIR}"
    log_info "Clone it first: git clone <your-repo> ${INSTALL_DIR}"
    exit 1
fi

# 2. Create reid.toml from template if missing
CONFIG_FILE="${INSTALL_DIR}/reid.toml"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_info "Creating reid.toml from template..."
    cat > "${CONFIG_FILE}" <<EOF
[immich]
# Immich API base URL (include port if not 80/443)
base_url = "http://localhost:2283"

# Read-scoped API key (assets:read, faces:read, people:read, search:read)
api_key = "CHANGE_ME_READ_KEY"

# Write-scoped API key (faces:write, people:write) — used only on confirm/promote
write_api_key = "CHANGE_ME_WRITE_KEY"

page_size = 2
timeout_seconds = 30

[store]
# SQLite database path (mounted at /data inside container)
path = "/data/reid.sqlite3"

[run]
log_level = "info"

[detection]
# Person detector thresholds (tuning knobs)
confidence_floor = 0.35
min_box_height = 100
pixel_source = "preview"  # "preview" or "original"

[clustering]
# Timespan window in days (default: 1 = same calendar day)
window_days = 1
# Cosine distance threshold for agglomerative clustering (0.5 default)
distance_threshold = 0.5
# Minimum cluster size for ask-for-name queue
min_cluster_size = 2
# Face-to-person containment ratio for anchor matching
face_overlap = 0.5

[head_detection]
# SCRFD head detector (fallback when Immich has no unassigned face).
# weights_path is optional: defaults to the weights vendored in the package.
# Uncomment to use your own file:
# weights_path = "/data/scrfd_500m.onnx"
input_size = 320
confidence_threshold = 0.4
top_expansion_ratio = 0.4
width_expansion_ratio = 0.2
nms_threshold = 0.45
EOF
    log_warn "Created ${CONFIG_FILE} — EDIT IT to add your Immich API keys!"
else
    log_info "reid.toml already exists, skipping."
fi

# 3. Create data directory
DATA_DIR="${INSTALL_DIR}/reid-data"
mkdir -p "${DATA_DIR}"
log_info "Data directory: ${DATA_DIR}"

# 4. Install systemd service files
SERVICE_DIR="/etc/systemd/system"
SERVICES=("reid-batch.service" "reid-batch.timer" "reid-ui.service")

for svc in "${SERVICES[@]}"; do
    SRC="${REPO_ROOT}/systemd/${svc}"
    DST="${SERVICE_DIR}/${svc}"
    if [[ -f "${SRC}" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would install ${SRC} -> ${DST}"
        else
            sudo cp "${SRC}" "${DST}"
            # Substitute variables in the installed file
            sudo sed -i "s|/home/pi/immich-person-reidentification|${INSTALL_DIR}|g" "${DST}"
            sudo sed -i "s|ghcr.io/loy/immich-person-reidentification:latest|${GHCR_IMAGE}|g" "${DST}"
            sudo sed -i "s|User=pi|User=${SERVICE_USER}|g" "${DST}"
            sudo sed -i "s|Group=pi|Group=${SERVICE_USER}|g" "${DST}"
            log_info "Installed ${svc}"
        fi
    else
        log_warn "Source not found: ${SRC}"
    fi
done

# 5. Reload systemd and enable
if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "Would run: systemctl daemon-reload"
    log_info "Would run: systemctl enable --now reid-batch.timer"
    log_info "Would run: systemctl enable --now reid-ui.service"
else
    log_info "Reloading systemd..."
    sudo systemctl daemon-reload

    log_info "Enabling reid-batch.timer (nightly at 02:00)..."
    sudo systemctl enable --now reid-batch.timer

    log_info "Enabling reid-ui.service (confirmation UI on :8080)..."
    sudo systemctl enable --now reid-ui.service

    log_info ""
    log_info "=== Installation complete ==="
    log_info ""
    log_info "Next steps:"
    log_info "  1. Edit ${CONFIG_FILE} with your Immich API keys"
    log_info "  2. Check status: systemctl status reid-ui"
    log_info "  3. View UI:      http://$(hostname -I | awk '{print $1}'):8080"
    log_info "  4. Run batch manually: systemctl start reid-batch"
    log_info "  5. View batch logs: journalctl -u reid-batch -f"
    log_info "  6. View UI logs:    journalctl -u reid-ui -f"
    log_info "  7. Check timer:     systemctl list-timers reid-batch.timer"
fi