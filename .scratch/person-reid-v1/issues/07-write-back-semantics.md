# Write-back semantics

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: open
Blocked by: [Immich API surface](01-immich-api-surface.md)

## Question

Exactly what happens on confirmation:

1. Which Immich API calls write the person label to which assets.
2. How dry-run records differ from real ones, and where dry-run results are kept for review before promoting to a real write.
3. Conflict handling: an asset already carries a different person label.
4. Idempotency: re-running nightly and re-confirming must not duplicate or corrupt state.
