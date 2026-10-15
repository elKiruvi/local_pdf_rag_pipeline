"""Unit checks for the generic publication guard without storing private examples."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest


def test_hygiene_scanner_flags_unwanted_content():
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    spec = importlib.util.spec_from_file_location("hygiene", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    findings = Counter()
    module.scan_content(b"safe text", Path("README.md"), findings)
    assert not findings
    module.scan_content(b"%PDF-1.0", Path("outputs/document.pdf"), findings)
    assert findings["binary"] == 1 and findings["forbidden_path"] == 1
    module.scan_content(b"person" + b"@" + b"example.test", Path("text.txt"), findings)
    assert findings["email"] == 1
    module.scan_content(b"marker", Path("text.txt"), findings, [b"marker"])
    assert findings["private_denylist"] == 1


def scanner():
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    spec = importlib.util.spec_from_file_location("hygiene", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,content,category",
    [
        ("examples/data.csv", b"a,b", "forbidden_path"),
        ("examples/notes.json", b"{}", "forbidden_path"),
        ("docs/notes.txt", b"notes", "forbidden_path"),
        ("README.md", b"%PDF-1.7\n", "binary"),
        (".github/workflows/credentials.yml", b"safe", "forbidden_path"),
        ("outputs/index/vectors.index", b"vector", "forbidden_path"),
        ("src/local_pdf_rag_pipeline/link.py", b"/home/" + b"example/Documents/file", "home_path"),
        ("README.md", b"\x00hidden", "binary"),
    ],
)
def test_publication_policy_rejects_synthetic_inputs(name, content, category):
    findings = Counter()
    scanner().scan_content(content, Path(name), findings)
    assert findings[category] > 0


def test_publication_policy_accepts_public_layout_and_detects_tokens():
    module = scanner()
    for name in (
        "README.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        ".github/dependabot.yml",
        "scripts/check_repository.py",
        "tests/test_hygiene.py",
        "docs/ARCHITECTURE.md",
        "src/local_pdf_rag_pipeline/cli.py",
        "examples/evaluation.json",
    ):
        findings = Counter()
        module.scan_content(b"Generic, independently authored text.\n", Path(name), findings)
        assert not findings, name
    assert not module.allowed(Path("docs/../README.md"))
    assert not module.allowed(Path("tests/private_notes.py"))
    findings = Counter()
    module.scan_content(b"ghp_" + b"a" * 36, Path("README.md"), findings)
    assert findings["service_token"] == 1
    findings = Counter()
    module.scan_content(b"password=" + b"x" * 20, Path("README.md"), findings)
    assert findings["credential_assignment"] == 1


def test_publication_policy_rejects_symlink_and_missing_files(tmp_path, monkeypatch):
    module = scanner()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "README.md").symlink_to(tmp_path / "target")
    findings = module.scan([Path("README.md"), Path("SECURITY.md")])
    assert findings["symlink_or_missing"] == 2


def test_history_scan_fails_closed_on_incomplete_git(monkeypatch):
    module = scanner()
    monkeypatch.setattr(module, "git", lambda *args: b"true")
    with pytest.raises(ValueError, match="Incomplete"):
        module.scan_history()
