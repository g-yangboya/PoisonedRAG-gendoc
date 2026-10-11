# Changelog

This repository is a derivative of
[PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (MIT License). It keeps
only the poisoned-document production pipeline and a single-configuration
acceptance check.

## [0.1.0] - 2026-10-11

### Added
- Structured poisoned-document generation pipeline:
  `gen_correct.py` -> `gen_incorrect.py` -> `gen_adv.py`, orchestrated by `run.py`.
- Per-stage traceability: `*.manifest.json`, `*.state.json` (resume) and
  `audit.jsonl` (per-query source mapping: BEIR split + FlashRAG record id/line).
- Behavior-validation loop in `gen_adv.py`: a body is regenerated until it
  induces the target answer, with a bounded number of trials and an explicit
  fallback that is recorded in the manifest.
- Frozen poisoned documents for NQ and HotpotQA: 1,000 queries x 5 adversarial
  bodies each (`results/pipeline_1000/`).
- Frozen Contriever retrieval results (`beir_results/`) used by the acceptance run.
- `ACCEPTANCE.md`: single-configuration ASR consistency check against the
  PoisonedRAG released 100-query set, using the `release_raw_substring_asr` metric.
- `sources.csv`: flat per-query mapping from each query to its source record and
  its five adversarial bodies.
- `LICENSE` (MIT), `CITATION.cff`, `requirements.txt` and `.gitignore`.

### Changed
- All input/output paths are parameterized; `run.py` now defaults to
  repository-relative paths instead of machine-specific absolute paths.
- CLI option names accept both `--snake_case` and `--kebab-case`.
- Manifests record repository-relative FlashRAG paths.
- The raw-substring ASR metric is reported as `release_raw_substring_asr`.
- Model configs use neutral placeholders (`OPENAI_API_KEY`, `/path/to/...`).

### Removed
- Copyrighted reference PDF, scratch page extracts, run logs and resumable
  state files from the published tree.
