"""Build traceable question + correct-answer (Q+C) sets for PoisonedRAG.

The implementation deliberately samples the complete BEIR query pool before doing
answer lookup.  It does not generate answers: every exported answer is either an
existing FlashRAG answer or an exact, contiguous substring of one.

这段代码用于为 PoisonedRAG 构建可追踪的 Question + Correct-answer(Q+C) 数据集。
它从 BEIR 查询池中采样问题，再去 FlashRAG 数据集中查找已有答案，只保留“能匹配到且只有一个有效参考答案”的问题。
代码不生成新答案，答案直接来自 FlashRAG 的 golden_answers。

最小运行：python gen_correct.py --dataset nq --revision bcafb8dd07d453be3cbeeeb3f78be1841bddf92c 
    主要输入来自命令行参数和环境变量: 
    必填: 
    --dataset: nq、hotpotqa、msmarco
    --revision: FlashRAG 的 HuggingFace revision，会解析为 commit
    可选，默认值如下: 
    --split test (msmarco 强制为 train)
    --cache-dir datasets/flashrag
    --output-dir results/qc/{dataset}
    --target 1000
    --initial-batch 1500
    --refill-batch 500
    --seed 100
    --rebuild-index: 重建 SQLite 索引
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RULE_VERSION = "qc-v1.2-selection-only"
INDEX_SCHEMA_VERSION = 2
HF_REPO = "RUC-NLPIR/FlashRAG_datasets"
NO_ANSWER_MARKERS = {"no answer present."}
DATASET_FILES = {
    "nq": ("nq/train.jsonl", "nq/dev.jsonl", "nq/test.jsonl"),
    "hotpotqa": ("hotpotqa/train.jsonl", "hotpotqa/dev.jsonl"),
    "msmarco": ("ms_marco/train.jsonl", "ms_marco/dev.jsonl"),
}


class QCError(RuntimeError):
    """A preparation or validation error that must stop the dataset run."""


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    answers: tuple[str, ...]
    source_file: str
    source_split: str
    source_line: int
    source_id: str


@dataclass
class AnswerIndex:
    exact: dict[str, list[AnswerRecord]]
    normalized: dict[str, list[AnswerRecord]]
    invalid_rows: list[dict[str, Any]]


class SQLiteAnswerIndex:
    """Read-only persistent exact/normalized FlashRAG question index."""

    def __init__(self, path: Path, *, reused: bool, record_count: int, invalid_count: int,
                 valid_answer_record_count: int, multiple_answer_record_count: int,
                 empty_answer_record_count: int):
        self.path = path
        self.reused = reused
        self.record_count = record_count
        self.invalid_count = invalid_count
        self.valid_answer_record_count = valid_answer_record_count
        self.multiple_answer_record_count = multiple_answer_record_count
        self.empty_answer_record_count = empty_answer_record_count
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)

    def lookup_exact(self, question: str) -> list[AnswerRecord]:
        return self._lookup("question", question)

    def lookup_normalized(self, question: str) -> list[AnswerRecord]:
        return self._lookup("normalized_question", normalize_question(question))

    def _lookup(self, column: str, value: str) -> list[AnswerRecord]:
        sql = (
            "SELECT question, answers_json, source_file, source_split, source_line, source_id "
            f"FROM answer_records WHERE {column} = ? ORDER BY source_order, source_line, row_id"
        )
        rows = self._connection.execute(sql, (value,)).fetchall()
        return [AnswerRecord(row[0], tuple(json.loads(row[1])), row[2], row[3], row[4], row[5]) for row in rows]

    def close(self) -> None:
        self._connection.close()


def canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def normalize_question(value: str) -> str:
    # Only NFC, case and whitespace normalization plus trailing question marks.
    return canonical_text(value).rstrip().rstrip("?").rstrip()


def normalize_answer(value: str) -> str:
    return canonical_text(value)


def is_no_answer(value: str) -> bool:
    return normalize_answer(value) in NO_ANSWER_MARKERS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _http_json(url: str, headers: Mapping[str, str] | None = None, timeout: float = 60) -> Any:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def find_cached_commit(dataset: str, revision: str, cache_dir: Path) -> str | None:
    """Resolve a revision from an already complete local cache without network access."""
    required = DATASET_FILES[dataset]

    # If --revision is already a commit SHA, use it directly when the cache is complete.
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        root = cache_dir / revision
        if all((root / relative).is_file() for relative in required):
            return revision

    if not cache_dir.exists():
        return None

    candidates: list[str] = []
    for version_path in cache_dir.glob("*/version.json"):
        try:
            metadata = json.loads(version_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(metadata, dict) or metadata.get("repo") != HF_REPO:
            continue

        commit = metadata.get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            continue

        root = version_path.parent
        if all((root / relative).is_file() for relative in required):
            candidates.append(commit)

    # A unique complete local cache is sufficient; no need to resolve the
    # symbolic revision (e.g. "main") against Hugging Face again.
    return candidates[0] if len(candidates) == 1 else None


def resolve_hf_revision(revision: str, token: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in HF_REPO.split("/"))
    url = f"https://huggingface.co/api/datasets/{quoted_repo}/revision/{urllib.parse.quote(revision, safe='')}"
    data = _http_json(url, headers)
    commit = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise QCError(f"Could not resolve Hugging Face revision {revision!r} to a commit")
    return commit


def _download(url: str, destination: Path, token: str | None, retries: int) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(retries + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            os.replace(temporary, destination)
            return
        except Exception as exc:  # download retries are explicitly permitted
            error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise QCError(f"Download failed for {url}: {error}") from error


def prepare_flashrag_files(
    dataset: str,
    revision: str,
    cache_dir: Path,
    *,
    token: str | None = None,
    retries: int = 2,
) -> tuple[str, list[dict[str, str]]]:
    if dataset not in DATASET_FILES:
        raise QCError(f"Unsupported dataset: {dataset}")
    # Prefer a complete local cache. Only contact Hugging Face when no
    # unambiguous usable local cache exists.
    commit = find_cached_commit(dataset, revision, cache_dir) or resolve_hf_revision(revision, token)
    root = cache_dir / commit
    version_path = root / "version.json"
    old = json.loads(version_path.read_text("utf-8")) if version_path.exists() else None
    if old is not None and old.get("repo") != HF_REPO:
        raise QCError(f"Cache repository mismatch in {version_path}")
    files: list[dict[str, str]] = []
    for relative in DATASET_FILES[dataset]:
        destination = root / relative
        known = next((item for item in (old or {}).get("files", []) if item.get("relative_path") == relative), None)
        if destination.exists() and known and sha256_file(destination) != known.get("sha256"):
            raise QCError(f"Cached file is corrupt: {destination}")
        if not destination.exists():
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{commit}/{relative}"
            _download(url, destination, token, retries)
        files.append({"relative_path": relative, "path": str(destination.resolve()), "sha256": sha256_file(destination)})
    # Preserve metadata for other datasets while making this dataset's files authoritative.
    merged = {item["relative_path"]: item for item in (old or {}).get("files", [])}
    merged.update({item["relative_path"]: item for item in files})
    atomic_json(version_path, {"repo": HF_REPO, "commit": commit, "files": list(merged.values())})
    return commit, files


def build_answer_index(files: Sequence[Mapping[str, str]], *, fail_on_invalid: bool = True) -> AnswerIndex:
    exact: dict[str, list[AnswerRecord]] = defaultdict(list)
    normalized: dict[str, list[AnswerRecord]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    for info in files:
        path = Path(info["path"])
        source_split = Path(info["relative_path"]).stem
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("row is not an object")
                    source_id, question, answers = row.get("id"), row.get("question"), row.get("golden_answers")
                    if not isinstance(source_id, str):
                        raise TypeError("id is not a string")
                    if not isinstance(question, str):
                        raise TypeError("question is not a string")
                    if not isinstance(answers, list) or any(not isinstance(item, str) for item in answers):
                        raise TypeError("golden_answers is not a string array")
                    cleaned = tuple(item.strip() for item in answers if item.strip() and not is_no_answer(item.strip()))
                    record = AnswerRecord(question, cleaned, str(path.resolve()), source_split, line_number, source_id)
                    exact[question].append(record)
                    normalized[normalize_question(question)].append(record)
                except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                    invalid.append({"file": str(path.resolve()), "line": line_number, "error": str(exc)})
    if invalid and fail_on_invalid:
        preview = "; ".join(f"{x['file']}:{x['line']}: {x['error']}" for x in invalid[:5])
        raise QCError(f"Found {len(invalid)} structurally invalid FlashRAG rows; {preview}")
    return AnswerIndex(dict(exact), dict(normalized), invalid)


def _index_signature(dataset: str, commit: str, files: Sequence[Mapping[str, str]]) -> str:
    return stable_json_hash({
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset": dataset,
        "commit": commit,
        "files": [{"relative_path": item["relative_path"], "sha256": item["sha256"]} for item in files],
    })


def _read_index_metadata(path: Path) -> dict[str, str]:
    try:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise QCError(f"SQLite answer index is corrupt: {path}; use --rebuild-index")
            return dict(connection.execute("SELECT key, value FROM index_metadata"))
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise QCError(f"Cannot read SQLite answer index {path}; use --rebuild-index: {exc}") from exc


def prepare_sqlite_answer_index(
    dataset: str,
    commit: str,
    files: Sequence[Mapping[str, str]],
    cache_dir: Path,
    *,
    fail_on_invalid: bool = True,
    rebuild: bool = False,
) -> SQLiteAnswerIndex:
    """Build once, then reuse a validated persistent FlashRAG lookup index."""
    index_dir = cache_dir / commit / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{dataset}.sqlite3"
    signature = _index_signature(dataset, commit, files)
    if index_path.exists() and not rebuild:
        metadata = _read_index_metadata(index_path)
        if metadata.get("signature") == signature:
            return SQLiteAnswerIndex(
                index_path, reused=True, record_count=int(metadata["record_count"]),
                invalid_count=int(metadata["invalid_count"]),
                valid_answer_record_count=int(metadata["valid_answer_record_count"]),
                multiple_answer_record_count=int(metadata["multiple_answer_record_count"]),
                empty_answer_record_count=int(metadata["empty_answer_record_count"]),
            )

    temporary = index_path.with_suffix(".sqlite3.building")
    temporary.unlink(missing_ok=True)
    print(f"Building persistent FlashRAG index: {index_path}", flush=True)
    connection = sqlite3.connect(temporary)
    invalid: list[dict[str, Any]] = []
    record_count = 0
    valid_answer_record_count = 0
    multiple_answer_record_count = 0
    empty_answer_record_count = 0
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE answer_records (
                row_id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_split TEXT NOT NULL,
                source_line INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                source_order INTEGER NOT NULL
            );
            CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        batch: list[tuple[Any, ...]] = []
        insert_sql = """INSERT INTO answer_records
            (question, normalized_question, answers_json, source_file, source_split,
             source_line, source_id, source_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        for source_order, info in enumerate(files):
            path = Path(info["path"])
            source_split = Path(info["relative_path"]).stem
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise TypeError("row is not an object")
                        source_id, question, answers = row.get("id"), row.get("question"), row.get("golden_answers")
                        if not isinstance(source_id, str):
                            raise TypeError("id is not a string")
                        if not isinstance(question, str):
                            raise TypeError("question is not a string")
                        if not isinstance(answers, list) or any(not isinstance(item, str) for item in answers):
                            raise TypeError("golden_answers is not a string array")
                        cleaned = [item.strip() for item in answers if item.strip() and not is_no_answer(item.strip())]
                        batch.append((question, normalize_question(question), json.dumps(cleaned, ensure_ascii=False),
                                      str(path.resolve()), source_split, line_number, source_id, source_order))
                        record_count += 1
                        if cleaned:
                            valid_answer_record_count += 1
                        else:
                            empty_answer_record_count += 1
                        if len(cleaned) > 1:
                            multiple_answer_record_count += 1
                        if len(batch) >= 5000:
                            connection.executemany(insert_sql, batch)
                            connection.commit()
                            batch.clear()
                    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                        invalid.append({"file": str(path.resolve()), "line": line_number, "error": str(exc)})
            print(f"Indexed {info['relative_path']}: {record_count} total records", flush=True)
        if batch:
            connection.executemany(insert_sql, batch)
        if invalid and fail_on_invalid:
            preview = "; ".join(f"{item['file']}:{item['line']}: {item['error']}" for item in invalid[:5])
            raise QCError(f"Found {len(invalid)} structurally invalid FlashRAG rows; {preview}")
        connection.executescript("""
            CREATE INDEX idx_answer_question ON answer_records(question);
            CREATE INDEX idx_answer_normalized_question ON answer_records(normalized_question);
            ANALYZE;
        """)
        metadata = {
            "schema_version": str(INDEX_SCHEMA_VERSION), "dataset": dataset, "commit": commit,
            "signature": signature, "record_count": str(record_count), "invalid_count": str(len(invalid)),
            "valid_answer_record_count": str(valid_answer_record_count),
            "multiple_answer_record_count": str(multiple_answer_record_count),
            "empty_answer_record_count": str(empty_answer_record_count),
            "files": json.dumps(list(files), ensure_ascii=False, sort_keys=True),
        }
        connection.executemany("INSERT INTO index_metadata(key, value) VALUES (?, ?)", metadata.items())
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    os.replace(temporary, index_path)
    print(f"Persistent FlashRAG index ready: {index_path} ({record_count} records)", flush=True)
    return SQLiteAnswerIndex(
        index_path, reused=False, record_count=record_count, invalid_count=len(invalid),
        valid_answer_record_count=valid_answer_record_count,
        multiple_answer_record_count=multiple_answer_record_count,
        empty_answer_record_count=empty_answer_record_count,
    )

def validate_queries(queries: Mapping[Any, Any] | Iterable[Mapping[str, Any]]) -> tuple[list[tuple[str, str]], list[str]]:
    rows = list(queries.items()) if isinstance(queries, Mapping) else [(row.get("id"), row.get("question")) for row in queries]
    seen: set[str] = set()
    valid: list[tuple[str, str]] = []
    blanks: list[str] = []
    for qid, question in rows:
        if not isinstance(qid, str) or not qid:
            raise QCError("Every BEIR query must have a non-empty string ID")
        if qid in seen:
            raise QCError(f"Duplicate BEIR query ID: {qid}")
        seen.add(qid)
        if not isinstance(question, str):
            raise QCError(f"BEIR query {qid!r} has a non-string question")
        if not question.strip():
            blanks.append(qid)
        else:
            valid.append((qid, question))
    return valid, blanks


def query_list_hash(rows: Sequence[tuple[str, str]]) -> str:
    return stable_json_hash([{"id": qid, "question": question} for qid, question in sorted(rows)])


def associate(question: str, index: AnswerIndex | SQLiteAnswerIndex) -> tuple[str | None, list[AnswerRecord]]:
    if isinstance(index, SQLiteAnswerIndex):
        records = index.lookup_exact(question)
        if records:
            return "exact", records
        records = index.lookup_normalized(question)
    else:
        records = index.exact.get(question)
        if records:
            return "exact", records
        records = index.normalized.get(normalize_question(question), [])
    return ("normalized", records) if records else (None, [])


def select_matched_answer(records: Sequence[AnswerRecord]) -> tuple[str, str | None, int, int]:
    """Accept exactly one answer; reject multi-answer annotations and source conflicts."""
    if not records:
        return "unmatched", None, 0, 0
    valid_records = [record for record in records if record.answers]
    if not valid_records:
        return "no_valid_answer", None, 0, 0
    all_answers = [answer for record in valid_records for answer in record.answers]
    normalized_answers = {normalize_answer(answer) for answer in all_answers}
    has_multi_answer_record = any(len(record.answers) > 1 for record in valid_records)
    if has_multi_answer_record or len(normalized_answers) > 1:
        return "multiple_answers", None, len(all_answers), len(normalized_answers)
    return "matched_single_answer", valid_records[0].answers[0], len(all_answers), 1


def _source_ref(record: AnswerRecord) -> dict[str, Any]:
    return {
        "split": record.source_split,
        "id": record.source_id,
        "line": record.source_line,
    }


def _load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("qid"), str):
                        result[row["qid"]] = row
                except json.JSONDecodeError as exc:
                    raise QCError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return result


def run_dataset(
    *,
    dataset: str,
    requested_split: str,
    queries: Mapping[Any, Any] | Iterable[Mapping[str, Any]],
    index: AnswerIndex | SQLiteAnswerIndex,
    output_dir: Path,
    flashrag_commit: str,
    flashrag_files: Sequence[Mapping[str, str]],
    selection_mode: str = "matched",
    index_info: Mapping[str, Any] | None = None,
    target: int = 1000,
    initial_batch: int = 1500,
    refill_batch: int = 500,
    seed: int = 100,
) -> dict[str, Any]:
    if selection_mode != "matched":
        raise QCError(f"Unsupported selection mode: {selection_mode}; only 'matched' is available")
    rows, blank_ids = validate_queries(queries)
    actual_split = "train" if dataset == "msmarco" else requested_split
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda item: item[0])
    rng.shuffle(ordered)
    qc_path, audit_path = output_dir / "qc.json", output_dir / "audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = _load_jsonl_by_qid(audit_path)
    qc: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter(blank_question=len(blank_ids))
    seen_questions: set[str] = set()
    cursor, batch_size = 0, initial_batch

    while cursor < len(ordered) and len(qc) < target:
        end = min(cursor + batch_size, len(ordered))
        counts["batch_candidates_taken"] += end - cursor
        for rank in range(cursor, end):
            if len(qc) >= target:
                break
            qid, question = ordered[rank]
            normalized = normalize_question(question)
            audit: dict[str, Any] = {"qid": qid, "sample_rank": rank}
            counts["actually_processed"] += 1
            if normalized in seen_questions:
                audit.update(status="rejected", stage="deduplicate", reason="duplicate_query", match_method=None)
            else:
                seen_questions.add(normalized)
                method, records = associate(question, index)
                audit.update(match_method=method, source_count=len(records))
                if records:
                    audit["source"] = _source_ref(records[0])
                match_reason, answer, answer_count, unique_answer_count = select_matched_answer(records)
                audit.update(answer_count=answer_count, unique_answer_count=unique_answer_count)
                if answer is None:
                    audit.update(status="rejected", stage="association", reason=match_reason)
                else:
                    counts["matched_with_single_answer"] += 1
                if answer is not None:
                    audit.update(status="accepted", stage="association", reason=match_reason, answer=answer)
                if audit.get("status") == "accepted":
                    assert answer is not None
                    qc[qid] = {"id": qid, "question": question, "correct answer": answer}
            counts[audit["reason"]] += 1
            # One audit row per actual processing attempt.  A prior completed run is
            # intentionally overwritten only by starting with a fresh output dir.
            if qid not in previous:
                append_jsonl(audit_path, audit)
            else:
                # Resume uses fresh deterministic reconstruction but avoids duplicate rows.
                pass
        cursor = end
        batch_size = refill_batch

    atomic_json(qc_path, qc)
    processed_count = counts["actually_processed"]
    multi_answer_filtered = counts["multiple_answers"]
    answer_bearing_matches = multi_answer_filtered + counts["matched_with_single_answer"]
    filtering_statistics = {
        "multiple_answer_filtered": multi_answer_filtered,
        "multiple_answer_share_of_processed": round(multi_answer_filtered / processed_count, 6) if processed_count else 0.0,
        "multiple_answer_share_of_answer_bearing_matches": round(multi_answer_filtered / answer_bearing_matches, 6) if answer_bearing_matches else 0.0,
    }
    manifest = {
        "dataset": dataset,
        "requested_beir_split": requested_split,
        "actual_beir_split": actual_split,
        "query_list_sha256": query_list_hash(rows),
        "flashrag": {"repo": HF_REPO, "commit": flashrag_commit, "files": list(flashrag_files)},
        "sampling": {"target": target, "initial_batch": initial_batch, "refill_batch": refill_batch, "seed": seed},
        "rule_version": RULE_VERSION,
        "selection_mode": selection_mode,
        "answer_selection_policy": "single_answer_only",
        "answer_index": dict(index_info or {}),
        "filtering_statistics": filtering_statistics,
        "counts": dict(counts | Counter(accepted=len(qc), query_pool=len(rows))),
        "target_reached": len(qc) >= target,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_beir_queries(dataset: str, split: str,
                      datasets_dir: Path | None = None) -> tuple[Mapping[str, str], str]:
    # Import lazily: importing src.utils also imports heavyweight retrieval packages.
    from src.utils import load_beir_datasets

    _corpus, queries, _qrels = load_beir_datasets(dataset, split, datasets_dir)
    return queries, "train" if dataset == "msmarco" else split


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate traceable Q+C files from BEIR and FlashRAG")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--beir_datasets_dir", "--beir-datasets-dir", dest="beir_datasets_dir",
        type=Path, default=None,
        help="BEIR datasets parent directory; the dataset name is appended automatically",
    )
    parser.add_argument("--revision", required=True, help="FlashRAG revision; resolved and pinned to a commit")
    parser.add_argument(
        "--cache_dir", "--cache-dir", dest="cache_dir", type=Path,
        default=Path("datasets/flashrag"))
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--initial_batch", "--initial-batch", dest="initial_batch", type=int, default=1500)
    parser.add_argument("--refill_batch", "--refill-batch", dest="refill_batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--selection_mode", "--selection-mode", dest="selection_mode",
        choices=["matched"], default="matched",
        help="Keep queries with exactly one valid matched reference answer (only supported mode)")
    parser.add_argument(
        "--rebuild_index", "--rebuild-index", dest="rebuild_index",
        action="store_true", help="Rebuild the persistent SQLite answer index")
    parser.add_argument(
        "--allow_invalid_answer_rows", "--allow-invalid-answer-rows",
        dest="allow_invalid_answer_rows", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:

    args = parse_args(argv)
    if min(args.target, args.initial_batch, args.refill_batch) <= 0:
        raise QCError("target and batch sizes must be positive")
    output_dir = args.output_dir or Path("results/qc") / args.dataset
    token = os.environ.get("HF_TOKEN")
    commit, files = prepare_flashrag_files(args.dataset, args.revision, args.cache_dir, token=token)
    index = prepare_sqlite_answer_index(
        args.dataset, commit, files, args.cache_dir,
        fail_on_invalid=not args.allow_invalid_answer_rows, rebuild=args.rebuild_index,
    )
    queries, _actual_split = load_beir_queries(
        args.dataset, args.split, args.beir_datasets_dir)
    # Record repository-relative paths in the manifest so it stays portable across
    # machines and free of local absolute paths. Internal file handles above keep
    # using the absolute paths for reading.
    portable_files = [
        {
            "relative_path": item["relative_path"],
            "path": (Path(commit) / item["relative_path"]).as_posix(),
            "sha256": item["sha256"],
        }
        for item in files
    ]
    manifest = run_dataset(
        dataset=args.dataset,
        requested_split=args.split,
        queries=queries,
        index=index,
        output_dir=output_dir,
        flashrag_commit=commit,
        flashrag_files=portable_files,
        selection_mode=args.selection_mode,
        index_info={
            "path": f"{commit}/indexes/{args.dataset}.sqlite3", "schema_version": INDEX_SCHEMA_VERSION,
            "signature": _index_signature(args.dataset, commit, files), "record_count": index.record_count,
            "valid_answer_record_count": index.valid_answer_record_count,
            "multiple_answer_record_count": index.multiple_answer_record_count,
            "empty_answer_record_count": index.empty_answer_record_count,
            "multiple_answer_share_of_all_records": round(index.multiple_answer_record_count / index.record_count, 6) if index.record_count else 0.0,
            "multiple_answer_share_of_valid_answer_records": round(index.multiple_answer_record_count / index.valid_answer_record_count, 6) if index.valid_answer_record_count else 0.0,
            "invalid_count": index.invalid_count, "reused": index.reused,
        },
        target=args.target,
        initial_batch=args.initial_batch,
        refill_batch=args.refill_batch,
        seed=args.seed,
    )
    index.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QCError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
