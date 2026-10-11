"""Small shared helpers for recoverable QCI generation stages."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping


class InputContractError(ValueError):
    pass


class StateMismatchError(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputContractError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise InputContractError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows, antivirus/indexing software can briefly hold the
        # destination open and make an otherwise valid atomic replace fail
        # with WinError 5/32. Retry the same atomic operation for a few
        # seconds; never fall back to an in-place write that could corrupt
        # resumable state.
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(min(0.05 * (2 ** attempt), 1.0))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _alias(record: Mapping[str, Any], qid: str, spaced: str, underscored: str, required: bool) -> str | None:
    first, second = record.get(spaced), record.get(underscored)
    if first is not None and second is not None and first != second:
        raise InputContractError(f"{qid}: conflicting fields {spaced!r} and {underscored!r}")
    value = first if first is not None else second
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputContractError(f"{qid}: {spaced!r} must be a non-empty string")
    return value


def validate_records(path: Path, require_incorrect: bool) -> dict[str, dict[str, str]]:
    data = load_json_strict(path)
    if not isinstance(data, dict):
        raise InputContractError(f"Top-level JSON must be an object: {path}")
    validated: dict[str, dict[str, str]] = {}
    for key, record in data.items():
        if not isinstance(key, str) or not key:
            raise InputContractError("Every top-level key must be a non-empty string")
        if not isinstance(record, dict):
            raise InputContractError(f"{key}: record must be an object")
        qid = record.get("id")
        if not isinstance(qid, str) or not qid or qid != key:
            raise InputContractError(f"{key}: record id must be the same non-empty string as its key")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise InputContractError(f"{key}: question must be a non-empty string")
        correct = _alias(record, key, "correct answer", "correct_answer", True)
        incorrect = _alias(record, key, "incorrect answer", "incorrect_answer", require_incorrect)
        item = {"id": qid, "question": question, "correct answer": correct}
        if incorrect is not None:
            item["incorrect answer"] = incorrect
        validated[key] = item
    return validated


def load_config(path: Path) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise InputContractError(f"Model config must be an object: {path}")
    return value


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_info": config.get("model_info"),
        "base_url": config.get("base_url"),
        "api_key_env": config.get("api_key_env"),
        "params": config.get("params", {}),
    }


def companion_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".state.json"), output.with_suffix(".manifest.json")


def ensure_resumable(output: Path, state_path: Path, signature: str) -> dict[str, Any] | None:
    if output.exists() and not state_path.exists():
        raise StateMismatchError(f"Output exists without verifiable state: {output}; use a new output path")
    if not state_path.exists():
        return None
    state = load_json_strict(state_path)
    if not isinstance(state, dict) or state.get("signature") != signature:
        raise StateMismatchError(f"State signature mismatch for {state_path}; use a new output path")
    return state


def response_public_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    return {key: response.get(key) for key in (
        "requested_model", "response_model", "finish_reason", "usage", "request_id", "system_fingerprint"
    ) if response.get(key) is not None}


def short_error(exc: Exception) -> dict[str, str]:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    reason = str(exc).replace("\n", " ")[:240]
    result = {"type": type(exc).__name__, "reason": reason}
    if status is not None:
        result["status"] = str(status)
    if code is not None:
        result["code"] = str(code)
    return result


def is_fatal_api_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {400, 401, 403, 404, 402}:
        return True
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("authentication", "permission", "notfound", "badrequest"))
