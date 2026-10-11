# PoisonedRAG-pro

Poisoned-document generation for knowledge-corruption attacks against
retrieval-augmented generation (RAG).

This repository builds question / correct-answer / incorrect-answer (Q/C/I)
records from BEIR and FlashRAG, then generates adversarial documents
("poisoned documents") that steer an answering model toward a target incorrect
answer. It is a derivative of
[PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (MIT License).

## Scope

This repository **only produces poisoned documents**. It ships:

- the generation pipeline: `gen_correct.py` -> `gen_incorrect.py` -> `gen_adv.py`,
  orchestrated by `run.py`;
- frozen poisoned documents for **NQ** and **HotpotQA**
  (1,000 queries x 5 adversarial bodies per query) under
  `results/pipeline_1000/`;
- frozen Contriever retrieval results under `beir_results/`;
- a single-configuration acceptance check that reuses `attack_main.py`
  (see `ACCEPTANCE.md`).

Out of scope (maintained in a companion artifact, not here): the paper's
normalized scorer `paper_normalized_final`, the full-grid offline
recomputation, the figures, and the perturbation / scoring audits.
MS-MARCO is an optional extension and is **not** part of the released panels.

## Pipeline

```text
BEIR queries + FlashRAG reference answers
        |
        v  gen_correct.py     -> qc.json   (Q + Correct), audit.jsonl, manifest.json
        v  gen_incorrect.py   -> qci.json  (Q + Correct + Incorrect)
        v  gen_adv.py         -> adv.json  (adv_texts: 5 bodies per query)
        v  run.py             -> chains the three stages
```

`gen_adv.py` validates each candidate body: the body alone is used as the
context to answer the original question, and it is regenerated until the proxy
answer matches the target incorrect answer (bounded number of trials, with an
explicit fallback recorded in the manifest).

### Scripts

| Script | Role | Key options |
|---|---|---|
| `gen_correct.py` | Sample queries and match a single FlashRAG reference answer | `--dataset`, `--revision`, `--target`, `--seed` |
| `gen_incorrect.py` | Generate a target incorrect answer per query | `--input`, `--output`, `--model_config_path` |
| `gen_adv.py` | Generate and validate adversarial bodies | `--qci_input_path`, `--adv_config_path`, `--adv_output_path`, `--adv_per_query` |
| `run.py` | Run the full pipeline for one or more datasets | `--datasets`, `--data_root`, `--output_root` |
| `evaluate_beir.py` | Compute BEIR retrieval results for a dataset | `--dataset`, `--model_code`, `--split` |
| `attack_main.py` | Acceptance evaluation (`release_raw_substring_asr`) | see `ACCEPTANCE.md` |

CLI options accept both `--snake_case` and `--kebab-case`.

## Data layout

```text
results/pipeline_1000/<dataset>/
    qc.json             Q + Correct
    qci.json            Q + Correct + Incorrect
    adv.json            + 5 adversarial bodies per query  (the release artifact)
    audit.jsonl         per-query source mapping (BEIR split + FlashRAG record id/line)
    manifest.json       dataset / FlashRAG revision / sampling / index metadata
    qci.manifest.json   incorrect-stage manifest
    adv.manifest.json   adv-stage manifest
beir_results/<dataset>-contriever.json   frozen retrieval results
sources.csv                              flat per-query source + body hashes
SHA256SUMS                               sha256 of every published file
```

## Reproduce

Two independent paths.

### 1. Offline acceptance (no API calls)

Needs BEIR datasets (`prepare_dataset.py`), the shipped `beir_results/`, local
Contriever weights, and local Llama-2-7b-chat weights. Set `local_path` in
`model_configs/llama-2-7b-chat_eval_config.json`, then:

```bash
python attack_main.py \
  --model_config_path model_configs/llama-2-7b-chat_eval_config.json \
  --all_adv_texts_path results/pipeline_1000/nq/adv.json \
  --eval_dataset nq --split test \
  --beir_datasets_dir ./datasets \
  --beir_results_dir ./beir_results \
  --retriever_model_path /path/to/contriever \
  --top_k 5 --adv_per_query 5 \
  --M 10 --repeat_times 100 \
  --name nq-llama2-acceptance
```

Expected numbers are in `ACCEPTANCE.md`.

### 2. API-based generation (needs an API key)

```bash
export OPENAI_API_KEY="your-api-key"        # name must match `api_key_env` in the config
python run.py --datasets nq,hotpotqa --data_root ./datasets
```

All input/output paths default to locations relative to the repository root;
override them with the CLI options (Windows and Linux use the same flags).

## Environment

- Python 3.10 (recommended).
- `pip install -r requirements.txt` — see the file for the offline / API split
  and optional backends.
- API generation reads the key from the environment variable named by
  `api_key_env` in the model config (default `OPENAI_API_KEY`).

## Metric naming

`attack_main.py` reports `release_raw_substring_asr`: a hit is counted when the
target incorrect answer appears as a substring of the model output. `Precision`,
`Recall` and `F1` are poisoned-document retrieval coverage, not BEIR relevance.
The paper's normalized scorer is `paper_normalized_final` and lives in the
companion artifact.

## Citation

If you use this repository, please cite the PoisonedRAG paper (see
`CITATION.cff`) and this resource.

## License

MIT License — see `LICENSE`. Portions of the code derive from PoisonedRAG,
which is distributed under the MIT License. BEIR, FlashRAG, Contriever and the
Llama-2 weights are external resources with their own licenses; download them
from their official sources under the applicable terms.
