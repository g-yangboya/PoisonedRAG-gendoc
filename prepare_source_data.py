"""Download BEIR source datasets used by query selection."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="nq,hotpotqa")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for dataset in (name.strip() for name in args.datasets.split(",")):
        if dataset not in {"nq", "hotpotqa", "msmarco"}:
            parser.error(f"unsupported dataset: {dataset}")
        target = args.data_root / dataset
        if not target.is_dir():
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            archive = args.data_root / f"{dataset}.zip"
            urlretrieve(url, archive)
            try:
                with ZipFile(archive) as zipped:
                    root = args.data_root.resolve()
                    for member in zipped.infolist():
                        destination = (root / member.filename).resolve()
                        if not destination.is_relative_to(root):
                            raise ValueError(f"Unsafe archive path: {member.filename}")
                    zipped.extractall(root)
            finally:
                archive.unlink(missing_ok=True)
        print(target)


if __name__ == "__main__":
    main()
