# Acceptance: ASR consistency check

This repository produces poisoned documents. Its acceptance evidence is a
**single-configuration** replay of the poisoning attack with the raw-substring
metric, compared against the PoisonedRAG released 100-query set. It is a
consistency check, **not** a full reproduction of the paper, and it does not
cover the paper's normalized scoring, full-grid recomputation, figures or
perturbation audits (those belong to a companion artifact).

## Configuration

| Item | Value |
|---|---|
| Answering model | `meta-llama/Llama-2-7b-chat-hf` (local, float16, eager) |
| Temperature | 0.1 |
| Max output tokens | 150 |
| Retriever | `contriever`, dot-product similarity |
| Injected bodies per query (`adv_per_query`) | 5 |
| Context size (`top_k`) | 5 |
| Queries per round (`M`) | 10 |
| Rounds (`repeat_times`) | 10 (PoisonedRAG 100q) / 100 (this repository, 1000q) |

The metric is `release_raw_substring_asr`: a query counts as a hit when the
target incorrect answer occurs as a substring of the model output. `Precision`,
`Recall` and `F1` below are **poisoned-document retrieval coverage**
(`a_i / k`, `a_i / N`, and their harmonic mean), not BEIR relevance measures.

## Results

| Dataset | Source | release_raw_substring_asr | P / R / F1 (coverage) |
|---|---|---|---|
| NQ | PoisonedRAG released 100 queries | 0.95 | 0.96 / 0.96 / 0.96 |
| NQ | this repository, 1000 queries | 0.92 | 0.98 / 0.98 / 0.98 |
| HotpotQA | PoisonedRAG released 100 queries | 0.98 | 1.00 / 1.00 / 1.00 |
| HotpotQA | this repository, 1000 queries | 0.93 | 1.00 / 1.00 / 1.00 |

The two sets use different query pools (100 vs 1000) and a single answering
model, so the numbers are consistent with, but not identical to, the paper.

## Reproduce (offline; no API calls)

Prerequisites:

- `datasets/{nq,hotpotqa}` in BEIR format (see `prepare_dataset.py`).
- `beir_results/{nq,hotpotqa}-contriever.json` (shipped in this repository).
- Local Contriever weights and local Llama-2-7b-chat weights. Set `local_path`
  in `model_configs/llama-2-7b-chat_eval_config.json` accordingly.

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

The final `release_raw_substring_asr (mean)` printed by the script corresponds
to the NQ row above. Repeat with `--eval_dataset hotpotqa` for the HotpotQA row.
