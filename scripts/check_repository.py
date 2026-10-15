"""Fail closed on common disclosures in tracked files and reachable local commits.

This is a publication guard, not a proof of provenance or a GitHub metadata audit.
Only category counts are printed; private patterns belong in an ignored local file.
"""

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
ROOT_FILES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
}
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".zip",
    ".7z",
    ".tar",
    ".npy",
    ".npz",
    ".index",
    ".faiss",
    ".exe",
    ".dll",
    ".pem",
    ".key",
    ".pyc",
    ".csv",
    ".txt",
    ".log",
    ".db",
    ".sqlite",
    ".bin",
    ".pkl",
}
FORBIDDEN_PARTS = {
    ".venv",
    "outputs",
    "data",
    "index",
    "quarantine",
    "private_audit",
    "corpus",
    "documents",
    "models",
    "logs",
    "credentials",
    "secrets",
}
PATTERNS = {
    "email": rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "home_path": rb"/(?:home|Users)/[A-Za-z0-9._-]+/",
    "windows_user_path": rb"[A-Za-z]:\\(?:Users|Documents and Settings)\\",
    "private_key": rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY",
    "credential_assignment": (
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
        rb"\s*[:=]\s*['\"]?[A-Za-z0-9_/-]{16,}"
    ),
    "service_token": rb"\b(?:gh[pousr]_|github_pat_|sk_live_|AKIA)[A-Za-z0-9_]{16,}\b",
}


def git(*args: str | bytes) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def allowed(relative: Path) -> bool:
    name = relative.as_posix()
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        return False
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(
            part.lower() in FORBIDDEN_PARTS
            or part.lower().startswith((".env", ".secret"))
            or "credential" in part.lower()
            or "private" in part.lower()
            for part in relative.parts
        )
    ):
        return False
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    parts = relative.parts
    return (
        name in ROOT_FILES
        or name == ".github/dependabot.yml"
        or (len(parts) == 3 and parts[:2] == (".github", "workflows") and name.endswith(".yml"))
        or (len(parts) == 2 and parts[0] == "docs" and name.endswith(".md"))
        or name == "examples/evaluation.json"
        or (len(parts) == 2 and parts[0] == "scripts" and name.endswith(".py"))
        or (
            len(parts) == 3
            and parts[:2] == ("src", "local_pdf_rag_pipeline")
            and name.endswith(".py")
        )
        or (
            len(parts) == 2
            and parts[0] == "tests"
            and parts[1].startswith("test_")
            and name.endswith(".py")
        )
    )


def tracked() -> list[Path]:
    return [Path(os.fsdecode(part)) for part in git("ls-files", "-z").split(b"\0") if part]


def scan_content(
    content: bytes, relative: Path, findings: Counter, extra_patterns: list[bytes] | None = None
) -> None:
    if not allowed(relative):
        findings["forbidden_path"] += 1
    if len(content) > MAX_BYTES:
        findings["oversized"] += 1
        return
    if (
        b"\0" in content
        or content.startswith((b"%PDF-", b"PK\x03\x04", b"\x7fELF", b"SQLite format 3"))
        or any(byte < 32 and byte not in (9, 10, 13) for byte in content)
    ):
        findings["binary"] += 1
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
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
        elif path.stat().st_size > MAX_BYTES:
            findings["oversized"] += 1
        else:
            scan_content(path.read_bytes(), relative, findings, extra_patterns)
    return findings


def scan_history(extra_patterns: list[bytes] | None = None) -> tuple[int, Counter]:
    """Inspect every reachable tree, including file modes and blobs removed from HEAD."""
    findings = Counter()
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ValueError("Incomplete Git history")
    commits = git("rev-list", "--all").splitlines()
    if not commits:
        raise ValueError("No reachable commits")
    seen = set()
    for commit in commits:
        for entry in git("ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
            if not entry:
                continue
            header, sep, raw_path = entry.partition(b"\t")
            fields = header.split()
            if not sep or len(fields) != 3:
                raise ValueError("Invalid Git tree entry")
            mode, kind, object_id = fields
            key = (object_id, raw_path, mode)
            if key in seen:
                continue
            seen.add(key)
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                findings["symlink_or_missing"] += 1
                continue
            relative = Path(os.fsdecode(raw_path))
            size = int(git("cat-file", "-s", object_id).strip())
            if size > MAX_BYTES:
                findings["oversized"] += 1
            else:
                scan_content(git("cat-file", "blob", object_id), relative, findings, extra_patterns)
    return len(seen), findings


def main() -> int:
    args = argparse.ArgumentParser(description="Scan tracked files without printing matches")
    args.add_argument(
        "--private-denylist", type=Path, help="Ignored local file, one regex per line"
    )
    args.add_argument("--history", action="store_true", help="Scan reachable local Git trees")
    options = args.parse_args()
    try:
        extra = []
        if options.private_denylist:
            if not options.private_denylist.is_file():
                raise ValueError("Missing local denylist")
            extra = [
                line
                for line in options.private_denylist.read_bytes().splitlines()
                if line.strip() and not line.startswith(b"#")
            ]
            if not extra:
                raise ValueError("Empty local denylist")
            for pattern in extra:
                re.compile(pattern)
            if options.private_denylist.resolve() in {
                (ROOT / path).resolve() for path in tracked()
            }:
                raise ValueError("Denylist must not be tracked")
        paths = tracked()
        if not paths:
            raise ValueError("Empty publication set")
        findings = scan(paths, extra)
        count = 0
        if options.history:
            count, older = scan_history(extra)
            findings.update(older)
        print(
            f"Publication set: {len(paths)} tracked files, {count} history entries; "
            f"category counts: {dict(findings)}"
        )
        return int(bool(findings))
    except (OSError, ValueError, subprocess.CalledProcessError, re.error):
        print("FAIL: publication audit incomplete or inaccessible (no details disclosed)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
