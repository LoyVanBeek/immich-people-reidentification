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
START="$(date +%s)"
FIRST="$(run)"
END="$(date +%s)"
echo "$FIRST" | tail -3
echo "$FIRST" | grep -q "assets fetched=" || { echo "FAIL: no summary in first run"; exit 1; }
echo "$FIRST" | grep -q "appearances embedded=" || { echo "FAIL: no appearance summary in first run"; exit 1; }
echo "$FIRST" | grep -q "gate drops: low_confidence=" || { echo "FAIL: no gate-drop summary in first run"; exit 1; }
# Coarse budget guard (ticket 10): the nightly batch must finish well within
# 30 minutes. The fixture set is tiny, so even under ARM64 emulation on x86
# this stays far below the limit; on the Pi itself it is minutes.
if [ $((END - START)) -ge 1800 ]; then
    echo "FAIL: first run took $((END - START))s, over the 30-minute budget"
    exit 1
fi

echo "== second run"
SECOND="$(run)"
echo "$SECOND" | tail -1
echo "$SECOND" | grep -q "assets fetched=0" || { echo "FAIL: second run reprocessed assets"; exit 1; }

echo "== wipe run"
WIPE="$(docker run --rm --platform linux/arm64 --network host \
    --entrypoint reid-wipe --user "$(id -u):$(id -g)" \
    -v "$WORK/reid.toml:/etc/reid/reid.toml:ro" \
    -v "$WORK:/data" \
    reid-batch:smoke --config /etc/reid/reid.toml)"
echo "$WIPE"
echo "$WIPE" | grep -q "store wiped: [1-9][0-9]* embeddings removed" \
    || { echo "FAIL: wipe did not remove embeddings"; exit 1; }

echo "== third run re-embeds after wipe"
THIRD="$(run)"
echo "$THIRD" | tail -1
echo "$THIRD" | grep -q "assets fetched=" || { echo "FAIL: no summary in third run"; exit 1; }
echo "$THIRD" | grep -q "appearances embedded=" || { echo "FAIL: wipe left no appearances to re-embed"; exit 1; }

echo "== OK: checkpointed feed, gate drops, and wipe hold across container runs"
