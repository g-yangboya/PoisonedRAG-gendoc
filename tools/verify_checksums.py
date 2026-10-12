"""Verify every file listed in SHA256SUMS without changing the release."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), f"missing: {relative}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, f"checksum mismatch: {relative}"
    print(f"Verified {len(lines)} release files")


if __name__ == "__main__":
    main()
