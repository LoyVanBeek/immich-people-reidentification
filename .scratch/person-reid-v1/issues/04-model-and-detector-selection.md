# Model and detector selection

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: open
Blocked by: [Re-ID model on ARM CPU](03-reid-model-arm-cpu.md)

## Question

Decide, with the research facts on the table:

1. Which Re-ID embedding model the container runs (OSNet variant vs alternatives), given Pi 5 CPU constraints.
2. Which person detector produces the full-person crops feeding it (nano YOLO-class or otherwise), its input resolution, and its throughput budget.
3. How crops failing quality checks (too small, clipped, low confidence) are handled.
