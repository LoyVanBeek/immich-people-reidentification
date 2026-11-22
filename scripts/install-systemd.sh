#!/usr/bin/env bash
# scripts/install-systemd.sh — Thin wrapper: install via systemd mode.
#
# Equivalent to: scripts/install.sh --mode systemd "$@"
exec "$(dirname "${BASH_SOURCE[0]}")/install.sh" --mode systemd "$@"
