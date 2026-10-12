"""Load only the BEIR source-query pool needed by resource construction."""

import csv
import json
from pathlib import Path


def load_beir_queries(dataset: str, split: str, datasets_dir: Path | None):
    if dataset not in {"nq", "hotpotqa", "msmarco"}:
        raise ValueError(f"Unsupported BEIR dataset: {dataset}")
    if dataset == "msmarco":
        split = "train"
    root = Path(datasets_dir or "datasets").expanduser().resolve()
    dataset_dir = root if root.name == dataset else root / dataset
    required = (dataset_dir / "queries.jsonl", dataset_dir / "qrels" / f"{split}.tsv")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BEIR source queries: {', '.join(missing)}; run prepare_source_data.py")
    all_queries = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            all_queries[record["_id"]] = record["text"]
    # Match BEIR GenericDataLoader: qrels select the split and determine ID order.
    queries = {}
    with (dataset_dir / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        next(rows)
        for row in rows:
            qid = row[0]
            if qid not in queries:
                queries[qid] = all_queries[qid]
    return queries
