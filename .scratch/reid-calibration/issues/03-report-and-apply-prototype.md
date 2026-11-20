# Report and apply prototype

Part of: [Re-ID calibration v1 — Map](../map.md)
Type: prototype
Status: open
Blocked by: 02

## Question

What should the calibration report and the apply flow look and behave like? Make a cheap, rough, concrete artifact to react to (fake data, no Immich needed), covering:

1. The per-knob report: suggested value, the data volume behind it, and the precision vs manual-touches curve with the human picking the point — both metrics side by side, per the effort's dual metric.
2. The gate spot-check queue: near-gate crops (just above/below each gate) with person / not-person verdicts.
3. The apply interaction inside the existing 3-pane workspace: where the report lives relative to the ⚙ dropdown, what applying a value does, and what revert/audit looks like if ticket 02 kept one.

Ticket 02 settled the data shape to design against: full per-knob curves persisted (candidate → metrics + volumes), a lexicographic highlight as the default pick, thin-data states before the cold-start volume minimums, labels-since-last-calibration as the staleness signal, multi-knob apply with a recalibrate hint, and one-click revert via the tuning history.

Prototype on a throwaway branch as the primary source, like v1's `prototype/confirmation-ui`. Run `prototype`.
