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


def scan_content(
    content: bytes, relative: Path, findings: Counter, extra_patterns: list[bytes] | None = None
) -> None:
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES or any(
        part in {".venv", "outputs", "data", "index", "quarantine", "private_audit"}
        for part in relative.parts
    ):
        findings["forbidden_path"] += 1
    if len(content) > 1_000_000:
        findings["oversized"] += 1
    if b"\0" in content or content.startswith((b"%PDF-", b"PK\x03\x04")):
        findings["binary"] += 1
    for label, pattern in PATTERNS.items():
        if re.search(pattern, content):
            findings[label] += 1
    if any(re.search(pattern, content, re.I) for pattern in extra_patterns or []):
        findings["private_denylist"] += 1


def scan(paths: list[Path], extra_patterns: list[bytes] | None = None) -> Counter:
    findings = Counter()
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            findings["symlink_or_missing"] += 1
            continue
        scan_content(path.read_bytes(), relative, findings, extra_patterns)
    return findings


def scan_history(extra_patterns: list[bytes] | None = None) -> tuple[int, Counter]:
    """Inspect every reachable blob, including content deleted from HEAD."""
    entries = subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=ROOT)
    findings = Counter()
    blob_count = 0
    for entry in entries.splitlines():
        object_id, _, path = entry.partition(b" ")
        if not path:
            continue
        kind = subprocess.check_output(["git", "cat-file", "-t", object_id], cwd=ROOT).strip()
        if kind != b"blob":
            continue
        blob_count += 1
        content = subprocess.check_output(["git", "cat-file", "blob", object_id], cwd=ROOT)
        scan_content(content, Path(os.fsdecode(path)), findings, extra_patterns)
    return blob_count, findings


def main() -> int:
    args = argparse.ArgumentParser(description="Scan tracked files without printing matches")
    args.add_argument(
        "--private-denylist", type=Path, help="Ignored local file, one regex per line"
    )
    args.add_argument("--history", action="store_true", help="Scan every reachable Git blob")
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
    count = 0
    if options.history:
        count, older = scan_history(extra)
        findings.update(older)
    print(
        f"Publication set: {len(paths)} tracked files, {count} history blobs; "
        f"category counts: {dict(findings)}"
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
