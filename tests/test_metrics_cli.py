import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_pdf_rag_pipeline.cli import main
from local_pdf_rag_pipeline.evaluate_retrieval import (
    evaluate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metrics_denominators_and_macro_averages():
    assert precision_at_k(["b", "a"], {"a", "c"}, 3) == pytest.approx(1 / 3)
    assert recall_at_k(["b", "a"], {"a", "c"}, 3) == pytest.approx(0.5)
    assert reciprocal_rank(["b", "a"], {"a", "c"}, 3) == pytest.approx(0.5)
    assert recall_at_k(["a"], set(), 1) == 0
    assert reciprocal_rank(["b"], {"a"}, 1) == 0
    cases = [
        {"question": "first", "relevant_ids": ["a"]},
        {"question": "second", "relevant_ids": ["z"]},
    ]
    result = evaluate(cases, lambda q, k: [{"id": "a"}] if q == "first" else [], 2)
    assert result == {
        "precision_at_k": 0.25,
        "recall_at_k": 0.5,
        "mrr_at_k": 0.5,
        "cases": 2,
        "k": 2,
    }
    for metric in (precision_at_k, recall_at_k, reciprocal_rank):
        with pytest.raises(ValueError, match="positive"):
            metric([], set(), 0)
    with pytest.raises(ValueError, match="At least"):
        evaluate([], lambda q, k: [], 3)


def test_cli_smoke_outside_project_root(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    assert main(["ingest", "--pdf", "outputs/demo.pdf"]) == 0
    assert main(["chunk"]) == 0
    chunks = json.loads((tmp_path / "outputs/chunks.json").read_text())
    assert all(item["source"] == "demo.pdf" for item in chunks)
    assert main(["retrieve", "private question"]) == 1
    captured = capsys.readouterr()
    assert "private question" not in captured.err
    assert "Error: ValueError" in captured.err
    assert main(["ingest", "--pdf", "missing.pdf"]) == 1
    assert "missing.pdf" not in capsys.readouterr().err


def test_packaged_entry_point_outside_project_root(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "local_pdf_rag_pipeline", "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Local PDF RAG pipeline" in result.stdout


def test_example_labels_match_newly_generated_chunks(tmp_path):
    assert main(["demo", "--output", str(tmp_path / "demo.pdf")]) == 0
    assert (
        main(
            [
                "ingest",
                "--pdf",
                str(tmp_path / "demo.pdf"),
                "--output",
                str(tmp_path / "pages.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "chunk",
                "--pages",
                str(tmp_path / "pages.json"),
                "--output",
                str(tmp_path / "chunks.json"),
            ]
        )
        == 0
    )
    chunks = json.loads((tmp_path / "chunks.json").read_text())
    cases = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/evaluation.json").read_text()
    )
    relevant = {case["question"]: case["relevant_ids"] for case in cases}
    by_id = {item["id"]: item for item in chunks}
    assert all(by_id[id_]["page"] == 2 for id_ in relevant["When should a backup be created?"])
    assert all(
        by_id[id_]["page"] == 1
        for id_ in relevant["What should happen after pressing the power button?"]
    )
