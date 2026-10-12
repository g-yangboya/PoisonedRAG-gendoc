# Changelog

## 0.2.1 — 2026-10-12

- Added field-level attribution, source-license links, and a change notice for redistributed question and reference-answer text; frozen data files are unchanged.

## 0.2.0 — 2026-10-12

- Scoped the release to frozen NQ and HotpotQA poisoning inputs and their construction pipeline.
- Moved the released panels to `resource/{nq,hotpotqa}/` without changing their JSON content.
- Removed acceptance evaluation, frozen retrieval outputs, local model backends, and their dependencies.
- Separated BEIR source-query loading from evaluation code and added a cross-platform source-data preparation command.
- Added record and checksum verification commands and clarified provenance, candidate ordering, and release limitations.

## 0.1.0 — 2026-10-11

- Initial release of generation scripts, provenance records, and frozen candidate passages.
