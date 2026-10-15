"""Unit checks for the generic publication guard without storing private examples."""

import importlib.util
from collections import Counter
from pathlib import Path


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
