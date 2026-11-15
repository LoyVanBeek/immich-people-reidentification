#!/usr/bin/env bash
# End-to-end smoke test for the Re-ID batch container.
#
# Builds the ARM64 image, starts a stub Immich HTTP server on the host, and
# runs the container twice: the first run fetches the fixture photos, the
# second run processes zero assets (checkpointed feed).
#
# Usage: scripts/container-smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORK="$(mktemp -d)"
IMAGE="reid-batch:smoke"
STUB_PORT=18283

cleanup() {
    [ -n "${STUB_PID:-}" ] && kill "$STUB_PID" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "== building $IMAGE for linux/arm64"
docker buildx build --platform linux/arm64 -t "$IMAGE" "$ROOT" >/dev/null

echo "== starting stub Immich on port $STUB_PORT"
uv run python -m tests.stub_immich --port "$STUB_PORT" &
STUB_PID=$!
sleep 1

cat > "$WORK/reid.toml" <<EOF
[immich]
base_url = "http://127.0.0.1:$STUB_PORT"
api_key = "reid-read-scoped-test-key"
page_size = 2

[store]
path = "/data/reid.sqlite3"

[run]
log_level = "info"
EOF

run() {
    docker run --rm --platform linux/arm64 --network host \
        --user "$(id -u):$(id -g)" \
        -v "$WORK/reid.toml:/etc/reid/reid.toml:ro" \
        -v "$WORK:/data" \
        reid-batch:smoke --config /etc/reid/reid.toml
}

echo "== first run"
FIRST="$(run)"
echo "$FIRST" | tail -3
echo "$FIRST" | grep -q "assets fetched=" || { echo "FAIL: no summary in first run"; exit 1; }

echo "== second run"
SECOND="$(run)"
echo "$SECOND" | tail -1
echo "$SECOND" | grep -q "assets fetched=0" || { echo "FAIL: second run reprocessed assets"; exit 1; }

echo "== OK: checkpointed feed holds across container runs"
