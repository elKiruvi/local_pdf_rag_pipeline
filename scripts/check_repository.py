"""Fail closed on common accidental disclosures in the publication set.

This generic guard is not a substitute for a local, private provenance audit.
Optional extra private patterns belong in an ignored local file, not in Git.
"""

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".zip",
    ".7z",
    ".npy",
    ".npz",
    ".index",
    ".faiss",
    ".exe",
    ".dll",
    ".pem",
    ".key",
    ".pyc",
}
PATTERNS = {
    "email": rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "home_path": rb"/(?:home|Users)/[A-Za-z0-9._-]+/",
    "windows_user_path": rb"[A-Za-z]:\\(?:Users|Documents and Settings)\\",
    "private_key": rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY",
    "credential_assignment": (
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|password)"
        rb"\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}"
    ),
}


def tracked() -> list[Path]:
    data = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [Path(os.fsdecode(part)) for part in data.split(b"\0") if part]


def scan(paths: list[Path], extra_patterns: list[bytes] | None = None) -> Counter:
    findings = Counter()
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            findings["symlink_or_missing"] += 1
            continue
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES or any(
            part in {".venv", "outputs", "data", "index", "quarantine", "private_audit"}
            for part in relative.parts
        ):
            findings["forbidden_path"] += 1
        if path.stat().st_size > 1_000_000:
            findings["oversized"] += 1
        content = path.read_bytes()
        if b"\0" in content or content.startswith((b"%PDF-", b"PK\x03\x04")):
            findings["binary"] += 1
        for label, pattern in PATTERNS.items():
            if re.search(pattern, content):
                findings[label] += 1
        if any(re.search(pattern, content, re.I) for pattern in extra_patterns or []):
            findings["private_denylist"] += 1
    return findings


def main() -> int:
    args = argparse.ArgumentParser(description="Scan tracked files without printing matches")
    args.add_argument(
        "--private-denylist", type=Path, help="Ignored local file, one regex per line"
    )
    options = args.parse_args()
    extra = []
    if options.private_denylist:
        extra = [
            line
            for line in options.private_denylist.read_bytes().splitlines()
            if line.strip() and not line.startswith(b"#")
        ]
    paths = tracked()
    if not paths:
        print("FAIL: no tracked files to audit")
        return 1
    findings = scan(paths, extra)
    print(f"Publication tree: {len(paths)} tracked files; category counts: {dict(findings)}")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
