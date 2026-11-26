# Docs

- [pipeline.md](pipeline.md) — how a run works: from a new Immich photo through
  detection, embedding, clustering, and anchoring to proposals, review, and
  the write path back to Immich.
- [data-model.md](data-model.md) — the local SQLite store: tables, keys,
  enums, and what survives a wipe.
- [diagrams/](diagrams/) — PlantUML sources (GitHub doesn't render PlantUML
  inline; view via PlantUML/Kroki):
  - [data-model.puml](diagrams/data-model.puml) — ER diagram of the local store
  - [pipeline.puml](diagrams/pipeline.puml) — swimlane diagram of a run and the write path
- [ui-learnings.md](ui-learnings.md) — user UI hints and decisions.
- [adr/](adr/) — architecture decision records.
- [agents/](agents/) — how engineering agents consume this repo's docs
  (domain model, issue tracker, triage labels).

Domain vocabulary lives in the root `GLOSSARY.md`.
