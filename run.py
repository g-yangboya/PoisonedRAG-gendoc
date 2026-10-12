"""Build Q/C/I records and adversarial documents for each dataset.

For each dataset the stages run sequentially: correct -> incorrect -> adv.
Datasets are processed sequentially and are selected with ``--datasets``
(default: ``nq,hotpotqa``). Existing compatible state files are resumed by the
generation scripts.

Every input and output path defaults to a location relative to the repository
root, so the pipeline does not depend on any machine-specific absolute path.
Override any of them with the CLI options below.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_DATASETS = ("nq", "hotpotqa", "msmarco")
FLASHRAG_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"


class StageIncomplete(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q/C/I records and adversarial documents for RAG poisoning.",
    )
    parser.add_argument(
        "--datasets", default="nq,hotpotqa",
        help="Comma-separated datasets to process (default: nq,hotpotqa).",
    )
    parser.add_argument(
        "--data_root", "--data-root", dest="data_root", type=Path,
        default=PROJECT_DIR / "datasets",
        help="Root directory containing the BEIR datasets (default: <repo>/datasets).",
    )
    parser.add_argument(
        "--flashrag_cache_dir", "--flashrag-cache-dir", dest="flashrag_cache_dir",
        type=Path, default=None,
        help="FlashRAG cache directory (default: <data_root>/flashrag).",
    )
    parser.add_argument(
        "--output_root", "--output-root", dest="output_root", type=Path,
        default=PROJECT_DIR / "generated_resource",
        help="Output root for qc/qci/adv files (default: <repo>/generated_resource).",
    )
    parser.add_argument(
        "--log_dir", "--log-dir", dest="log_dir", type=Path,
        default=PROJECT_DIR / "logs",
        help="Directory for per-stage logs (default: <repo>/logs).",
    )
    parser.add_argument(
        "--target", type=int, default=1000,
        help="Number of queries to build per dataset.",
    )
    parser.add_argument(
        "--adv_per_query", "--adv-per-query", dest="adv_per_query",
        type=int, default=5,
        help="Number of adversarial documents to generate per query.",
    )
    parser.add_argument(
        "--revision", default=FLASHRAG_REVISION,
        help="Pinned FlashRAG dataset revision.",
    )
    parser.add_argument(
        "--seed", type=int, default=100,
        help="Sampling seed passed to gen_correct.py.",
    )
    parser.add_argument(
        "--incorrect_config", "--incorrect-config", dest="incorrect_config",
        type=Path,
        default=PROJECT_DIR / "model_configs" / "gpt-5-mini_incorrect_config.json",
        help="Model config used by gen_incorrect.py.",
    )
    parser.add_argument(
        "--adv_config", "--adv-config", dest="adv_config",
        type=Path,
        default=PROJECT_DIR / "model_configs" / "gpt-5-mini_adv_config.json",
        help="Model config used by gen_adv.py.",
    )
    args = parser.parse_args(argv)
    args.datasets = tuple(
        name.strip() for name in str(args.datasets).split(",") if name.strip()
    )
    if not args.datasets:
        parser.error("at least one dataset must be provided via --datasets")
    unknown = [name for name in args.datasets if name not in SUPPORTED_DATASETS]
    if unknown:
        parser.error(
            f"unsupported dataset(s): {', '.join(unknown)}; "
            f"choose from {', '.join(SUPPORTED_DATASETS)}"
        )
    if args.flashrag_cache_dir is None:
        args.flashrag_cache_dir = args.data_root / "flashrag"
    return args


def argument_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def report_incomplete(dataset: str, stage: str, command: list[str]) -> None:
    output_flag = "--output" if stage == "incorrect" else "--adv_output_path"
    output_value = argument_value(command, output_flag)
    unfinished: list[str] = []
    if output_value:
        manifest_path = Path(output_value).with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest = load_object(manifest_path)
            unfinished = manifest.get("unfinished_ids", [])

    supplemental = list(command)
    if "--request_seed_offset" in supplemental:
        index = supplemental.index("--request_seed_offset")
        supplemental[index + 1] = "1"
    else:
        supplemental.extend(["--request_seed_offset", "1"])

    print(
        f"[{dataset}] {stage}: {len(unfinished)} record(s) remain incomplete "
        f"after retries.",
        flush=True,
    )
    if unfinished:
        print("Unfinished IDs: " + ", ".join(unfinished), flush=True)
    print("Run this supplemental experiment once with a different API seed:", flush=True)
    print(subprocess.list2cmdline(supplemental), flush=True)
    print(
        "Then rerun run.py. If the supplemental command still returns 3, "
        "repeat it with --request_seed_offset 2.",
        flush=True,
    )


def run_stage(dataset: str, stage: str, command: list[str],
              log_dir: Path, incomplete_retries: int = 1) -> None:
    """Run one stage synchronously and save a flat per-stage log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline-{dataset}-{stage}.log"
    for attempt in range(1, incomplete_retries + 1):
        print(
            f"[{dataset}] {stage}: started (pass {attempt}/{incomplete_retries}); "
            f"log={log_path}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write(f"PASS: {attempt}/{incomplete_retries}\n")
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            print(f"[{dataset}] {stage}: complete", flush=True)
            return
        if completed.returncode == 3 and attempt == incomplete_retries:
            report_incomplete(dataset, stage, command)
            raise StageIncomplete(f"{dataset} {stage} remains incomplete")
        if completed.returncode != 3:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(
            f"[{dataset}] {stage}: incomplete; retrying unfinished records",
            flush=True,
        )


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Expected output was not created: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def require_count(path: Path, expected: int, stage: str) -> dict[str, Any]:
    records = load_object(path)
    if len(records) != expected:
        raise RuntimeError(
            f"{stage} output has {len(records)} records; expected {expected}: {path}"
        )
    return records


def validate_adv(path: Path, expected: int, adv_per_query: int) -> None:
    records = require_count(path, expected, "adv")
    invalid = [
        qid for qid, record in records.items()
        if not isinstance(record, dict)
        or not isinstance(record.get("adv_texts"), list)
        or len(record["adv_texts"]) != adv_per_query
        or any(not isinstance(text, str) or not text.strip() for text in record["adv_texts"])
    ]
    if invalid:
        preview = ", ".join(invalid[:10])
        raise RuntimeError(
            f"adv output contains {len(invalid)} invalid records; first IDs: {preview}"
        )


def paths_for(output_root: Path, dataset: str) -> tuple[Path, Path, Path, Path]:
    dataset_dir = output_root / dataset
    return (
        dataset_dir,
        dataset_dir / "qc.json",
        dataset_dir / "qci.json",
        dataset_dir / "adv.json",
    )


def run_dataset(dataset: str, args: argparse.Namespace) -> None:
    output_dir, qc_path, qci_path, adv_path = paths_for(args.output_root, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stage(dataset, "correct", [
        sys.executable, "-u", str(PROJECT_DIR / "gen_correct.py"),
        "--dataset", dataset,
        "--split", "train" if dataset == "msmarco" else "test",
        "--beir_datasets_dir", str(args.data_root),
        "--revision", args.revision,
        "--cache_dir", str(args.flashrag_cache_dir),
        "--output_dir", str(output_dir),
        "--target", str(args.target),
        "--seed", str(args.seed),
    ], log_dir=args.log_dir)
    require_count(qc_path, args.target, "correct")

    run_stage(dataset, "incorrect", [
        sys.executable, "-u", str(PROJECT_DIR / "gen_incorrect.py"),
        "--input", str(qc_path),
        "--output", str(qci_path),
        "--model_config_path", str(args.incorrect_config),
        "--response_retries", "2",
        "--retry_pending",
    ], log_dir=args.log_dir)
    require_count(qci_path, args.target, "incorrect")

    run_stage(dataset, "adv", [
        sys.executable, "-u", str(PROJECT_DIR / "gen_adv.py"),
        "--qci_input_path", str(qci_path),
        "--adv_config_path", str(args.adv_config),
        "--adv_output_path", str(adv_path),
        "--adv_per_query", str(args.adv_per_query),
        "--response_retries", "2",
        "--retry_pending",
    ], log_dir=args.log_dir)
    validate_adv(adv_path, args.target, args.adv_per_query)
    print(
        f"[{dataset}] verified: {args.target} queries x "
        f"{args.adv_per_query} adversarial documents",
        flush=True,
    )


def preflight(args: argparse.Namespace) -> None:
    required_files = [
        args.incorrect_config,
        args.adv_config,
        PROJECT_DIR / "gen_correct.py",
        PROJECT_DIR / "gen_incorrect.py",
        PROJECT_DIR / "gen_adv.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if not args.data_root.is_dir():
        missing.append(str(args.data_root))
    else:
        missing.extend(
            str(args.data_root / dataset)
            for dataset in args.datasets
            if not (args.data_root / dataset).is_dir()
        )
    if missing:
        raise FileNotFoundError("Missing pipeline input(s): " + ", ".join(missing))
    for config_path in (args.incorrect_config, args.adv_config):
        config = load_object(config_path)
        key_env = config.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            raise RuntimeError(
                f"Environment variable {key_env!r} required by {config_path.name} is not set"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight(args)
    try:
        for dataset in args.datasets:
            run_dataset(dataset, args)
    except StageIncomplete:
        return 3
    print(
        f"Pipeline complete: {len(args.datasets)} dataset(s), "
        f"{args.target * len(args.datasets)} queries, "
        f"{args.target * args.adv_per_query * len(args.datasets)} adversarial documents."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
