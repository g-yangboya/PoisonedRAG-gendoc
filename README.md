# A Reusable Frozen Poisoning Resource

This repository releases the frozen poisoning resource used in the paper and the pipeline used to construct it. It does not contain the paper's defense evaluation or analysis code.

## Released resource

`resource/nq/` and `resource/hotpotqa/` each contain 1,000 selected query records and five **ordered** malicious passage bodies per query: 2,000 queries and 10,000 bodies in total. Each query has a reference answer and a fixed attacker target. The candidates were generated independently of the evaluated defenses and were not regenerated in response to their outputs.

The files are shared attack inputs for paired comparisons, not a benchmark leaderboard or a guarantee that a candidate succeeds against any particular system. The NQ panel is not a simple superset of the public PoisonedRAG 100-query artifact: only 42 questions overlap. Absolute results from the two panels should not be compared as if only sample size changed.

## Directory layout and records

```text
resource/{nq,hotpotqa}/
  qc.json             query ID -> id, question, correct answer
  qci.json            qc fields + incorrect answer (fixed attacker target)
  adv.json            qci fields + adv_texts (array of five ordered bodies)
  audit.jsonl         query-selection decisions and accepted source mappings
  manifest.json       source revision, selection policy, counts, index metadata
  qci.manifest.json   target-generation stage, prompt version, counts, usage
  adv.manifest.json   passage-generation stage, validation totals, usage
sources.csv            one row per query with source mapping and body hashes
SHA256SUMS             SHA-256 digest of every released file except itself
DATA_ATTRIBUTION.md    field-level source attribution and change notice
```

The outer JSON keys are stable query IDs. Within each `adv_texts` array, index 0 is candidate 1; budget `m` uses `adv_texts[:m]` for `m=0,...,5`. Do not sort these arrays. `sources.csv` records the hash of each body at positions 1–5. Candidate order is therefore independently checkable. The released `adv.manifest.json` records dataset-level validated/fallback totals; it does not identify individual fallback bodies.

For NQ, generation-time checks validated 4,777 bodies and retained 223 fallback bodies. For HotpotQA, the counts are 4,770 and 230. Exhausting the bounded proxy-validation trials retained a nonempty final body. These generation-time checks do not establish semantic quality or success against the evaluated generators and defenses.

The query-selection `manifest.json` pins the FlashRAG revision and records the BEIR split, sampling seed and batches, filtering counts, selection policy, and answer-index signature. `audit.jsonl` supplies per-query FlashRAG source record, split, and line for accepted entries. The two stage manifests identify prompt versions and input signatures, request and token totals, unfinished IDs, and aggregate validation outcomes. Generation scripts contain the prompts and model settings. API-side model snapshots and stochastic responses can prevent byte-identical regeneration.

## Constructing the resource

The three stages are **query/reference selection** (`gen_correct.py`), **attacker-target generation** (`gen_incorrect.py`), and **poisoned-passage generation plus proxy validation** (`gen_adv.py`). `run.py` orchestrates them. These script names are retained for compatibility; “incorrect” means a fixed answer incompatible with the reference, not a random label. The implementation uses an OpenAI-compatible API backend and reads its key from the environment variable named by `api_key_env` in the generation configs.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python prepare_source_data.py --datasets nq,hotpotqa
# Set OPENAI_API_KEY in your environment before this step.
python run.py --datasets nq,hotpotqa --data_root ./datasets --output_root ./generated_resource
```

`gen_correct.py` downloads the pinned FlashRAG source records on first use. By default `run.py` targets 1,000 accepted records and five candidates per query in `generated_resource/`. A fresh API regeneration should use a new output directory: the released JSON has no resumable `*.state.json`, and the scripts intentionally reject an existing output without matching state. Re-execution is therefore a reconstruction path, not a promise of byte-identical output.

## Verify the frozen release

```bash
python tools/verify_resource.py
python tools/verify_checksums.py
```

The first command checks IDs, field continuity, 1,000 × 5 cardinality for each panel, and body hashes in `sources.csv`. The second checks `SHA256SUMS`. `tools/build_release_artifacts.py` rebuilds `sources.csv` and then `SHA256SUMS` after an intentional release change; do not run it during verification.

## Intended use and limits

Use these frozen candidates for controlled RAG poisoning and defense research with shared inputs. The release does not include retrieval outputs, paper scoring, figures, or analysis code. Different generators, retrievers, prompts, and scoring rules define different evaluation conditions and clean gain sets. These candidates were not optimized against evaluated defenses, so the resource does not support adaptive-attack robustness claims or a system-independent leaderboard.

## Provenance, citation, and licenses

The code is a derivative of [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (MIT). Cite both that work and this resource; see `CITATION.cff`. The `LICENSE` file applies to this repository's original code and any MIT-licensed derived code. It does **not** relicense upstream question or reference-answer text. [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) maps released fields to Natural Questions, HotpotQA, and FlashRAG, names their source licenses, and describes the selection and transformation of those fields. Retain those attributions and observe the applicable upstream terms when reusing the data.
