"""Rebuild the release tables: ``sources.csv`` and ``SHA256SUMS``.

Reads the frozen artifacts under ``resource``
and writes two repository-root files:

* ``sources.csv``  -- per-query source mapping and the sha256 of each body.
* ``SHA256SUMS``   -- sha256 of every published file (for release pinning).

Usage::

    python tools/build_release_artifacts.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "resource"
DATASETS = ("nq", "hotpotqa")

EXCLUDE_TOP_DIRS = {"tmp", "logs", "datasets", "models", "doc", "results", "beir_results", "generated_resource", ".venv", "venv", "env"}
EXCLUDE_ANY_DIRS = {".git", "__pycache__", ".pytest_cache"}
EXCLUDE_NAMES = {"SHA256SUMS", "操作指南.md", "验收情况.md"}
EXCLUDE_PREFIXES = ("resource/msmarco/",)
EXCLUDE_SUFFIXES = (".state.json", ".pyc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accepted_sources(audit_path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if not audit_path.is_file():
        return mapping
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "accepted" and row.get("qid"):
                mapping[row["qid"]] = row.get("source") or {}
    return mapping


def build_sources() -> int:
    columns = [
        "dataset", "query_id", "beir_split", "flashrag_revision",
        "source_split", "source_record_id", "source_line",
        "question", "correct_answer", "incorrect_answer",
        "adv_1_sha256", "adv_2_sha256", "adv_3_sha256", "adv_4_sha256", "adv_5_sha256",
    ]
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        directory = RESULTS / dataset
        adv = json.loads((directory / "adv.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        revision = manifest.get("flashrag", {}).get("commit", "")
        split = manifest.get("actual_beir_split", manifest.get("requested_beir_split", ""))
        sources = load_accepted_sources(directory / "audit.jsonl")
        for qid, record in adv.items():
            source = sources.get(qid, {})
            bodies = record.get("adv_texts", [])
            hashes = [hashlib.sha256(body.encode("utf-8")).hexdigest() for body in bodies]
            hashes += [""] * (5 - len(hashes))
            rows.append({
                "dataset": dataset,
                "query_id": qid,
                "beir_split": split,
                "flashrag_revision": revision,
                "source_split": source.get("split", ""),
                "source_record_id": source.get("id", ""),
                "source_line": source.get("line", ""),
                "question": record.get("question", ""),
                "correct_answer": record.get("correct answer", ""),
                "incorrect_answer": record.get("incorrect answer", ""),
                "adv_1_sha256": hashes[0],
                "adv_2_sha256": hashes[1],
                "adv_3_sha256": hashes[2],
                "adv_4_sha256": hashes[3],
                "adv_5_sha256": hashes[4],
            })
    with (ROOT / "sources.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_hashes() -> int:
    lines: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(ROOT).parts
        if parts[0] in EXCLUDE_TOP_DIRS or (set(parts) & EXCLUDE_ANY_DIRS):
            continue
        if relative in EXCLUDE_NAMES:
            continue
        if relative.startswith(EXCLUDE_PREFIXES):
            continue
        if relative.endswith(EXCLUDE_SUFFIXES):
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    (ROOT / "SHA256SUMS").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return len(lines)


def main() -> None:
    print(f"sources.csv: {build_sources()} rows")
    print(f"SHA256SUMS: {build_hashes()} files")


if __name__ == "__main__":
    main()
