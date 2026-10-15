"""Unit checks for the generic publication guard without storing private examples."""

import importlib.util
import shutil
import subprocess
import sys
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


def test_publication_cli_rejects_forced_add_without_exposing_matches(tmp_path):
    repository = tmp_path / "synthetic-publication"
    (repository / "scripts").mkdir(parents=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "scripts/check_repository.py",
        repository / "scripts/check_repository.py",
    )
    fake_token = "ghp_" + "A7sT4vR9kL3mN6pQ2wX8yZ5bC1dE0fG4hJ7i"
    samples = {
        "examples/forbidden.csv": "field,value\n",
        "examples/notes.json": "{}\n",
        "docs/notes.txt": "text\n",
        "docs/renamed.md": "%PDF-1.7\n",
        ".env": "sample\n",
        "outputs/index/vectors.index": "document-derived index\n",
        "tests/test_credentials.py": "api_key: " + fake_token + "\n",
    }
    for name, content in samples.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(repository)], check=True, capture_output=True)
    subprocess.run(["git", "add", "-f", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=test" + "@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Synthetic negative case",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, str(repository / "scripts/check_repository.py"), "--history"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert all(
        category in result.stdout for category in ("forbidden_path", "binary", "service_token")
    )
    assert fake_token not in result.stdout + result.stderr
    assert all(name not in result.stdout + result.stderr for name in samples)
