"""Offline contract tests using newly authored text and fake Ollama responses."""

import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pymupdf
import pytest

from local_pdf_rag_pipeline.build_index import build_index, load_index
from local_pdf_rag_pipeline.chunker import chunk_pages, line_kind
from local_pdf_rag_pipeline.demo import create_demo
from local_pdf_rag_pipeline.embed import LOCAL_OLLAMA_HOST, embed_texts, local_client
from local_pdf_rag_pipeline.ingest import clean_text, extract_pages
from local_pdf_rag_pipeline.rag import SYSTEM_PROMPT, answer, assemble_messages
from local_pdf_rag_pipeline.retrieve import retrieve


class FakeClient:
    def __init__(self, embeddings=None, response="The backup is on Friday."):
        self.embeddings = [[3.0, 4.0]] if embeddings is None else embeddings
        self.response = response
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        return SimpleNamespace(embeddings=self.embeddings)

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return SimpleNamespace(message=SimpleNamespace(content=self.response))


def chunks_fixture():
    return [
        {
            "id": "chunk-0000",
            "text": "Power turns on the device.",
            "source": "demo.pdf",
            "page": 1,
            "section": "START",
            "content_type": "text",
        },
        {
            "id": "chunk-0001",
            "text": "Backup on Friday.",
            "source": "demo.pdf",
            "page": 2,
            "section": "BACKUP",
            "content_type": "text",
        },
    ]


def test_clean_and_extract(tmp_path):
    assert clean_text("Re-\nconnect  now\r\n\tNext\u00ad line") == "Reconnect now\nNext line"
    path = create_demo(tmp_path / "demo.pdf")
    pages = extract_pages(path)
    assert [item["page"] for item in pages] == [1, 2]
    assert all(item["source"] == "demo.pdf" and item["text"] for item in pages)
    with pytest.raises(ValueError, match="existing PDF"):
        extract_pages(tmp_path / "missing.pdf")
    (tmp_path / "malformed.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="Unable to extract"):
        extract_pages(tmp_path / "malformed.pdf")
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")
    blank.close()
    with pytest.raises(ValueError, match="no extractable text"):
        extract_pages(tmp_path / "blank.pdf")


def test_chunk_boundaries_and_heuristics():
    assert line_kind("# Overview") == "heading"
    assert line_kind("2.1 Setup") == "heading"
    assert line_kind("a | b | c") == "table"
    assert line_kind("ordinary words") == "text"
    pages = [
        {"source": "demo.pdf", "page": 1, "text": "# Setup\n" + "a" * 13 + "\nx | y | z"},
        {"source": "demo.pdf", "page": 2, "text": "# Next\nhello"},
    ]
    chunks = chunk_pages(pages, size=10, overlap=2)
    assert [(c["page"], c["content_type"]) for c in chunks] == [
        (1, "text"),
        (1, "text"),
        (1, "table"),
        (2, "text"),
    ]
    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]
    assert chunks[0]["section"] == "# Setup"
    assert chunks[-1]["section"] == "# Next"
    assert [c["id"] for c in chunks] == [f"chunk-{i:04d}" for i in range(4)]
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(pages, size=10, overlap=10)
    with pytest.raises(ValueError, match="No text"):
        chunk_pages([{"source": "demo.pdf", "page": 1, "text": "# Empty"}])
    with pytest.raises(ValueError, match="Pages require"):
        chunk_pages([{"source": "", "page": 0, "text": "hi"}])


def test_embedding_shape_normalization_and_errors():
    client = FakeClient(embeddings=[[3, 4], [0, 5]])
    vectors = embed_texts(["one", "two"], model="local-model", client=client)
    assert vectors.dtype == np.float32 and vectors.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1])
    assert client.calls[0][1] == {"model": "local-model", "input": ["one", "two"]}
    for bad in ([], [[0, 0]], [[float("nan"), 1]], [[1, 2], [1]]):
        with pytest.raises(ValueError):
            embed_texts(["one"], client=FakeClient(embeddings=bad))
    with pytest.raises(ValueError, match="nonempty"):
        embed_texts([""])

    class Offline:
        def embed(self, **kwargs):
            raise ConnectionError("private connection details")

    with pytest.raises(RuntimeError, match="Local Ollama"):
        embed_texts(["one"], client=Offline())


def test_local_ollama_client_ignores_environment_host(monkeypatch):
    settings = {}

    def capture_client(**kwargs):
        settings.update(kwargs)
        return FakeClient()

    monkeypatch.setenv("OLLAMA_HOST", "https://remote.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("OLLAMA_API_KEY", "dummy")
    monkeypatch.setattr("local_pdf_rag_pipeline.embed.Client", capture_client)
    assert isinstance(local_client(), FakeClient)
    assert settings == {
        "host": LOCAL_OLLAMA_HOST,
        "headers": {"Authorization": ""},
        "trust_env": False,
    }


def test_index_retrieval_alignment_and_ranking(tmp_path):
    chunks = chunks_fixture()
    vectors = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="match"):
        build_index(vectors[:1], chunks, tmp_path / "bad")
    with pytest.raises(ValueError, match="L2-normalized"):
        build_index(vectors * 2, chunks, tmp_path / "bad")
    with pytest.raises(ValueError, match="No index"):
        load_index(tmp_path / "missing")
    build_index(vectors, chunks, tmp_path / "index")
    index, metadata = load_index(tmp_path / "index")
    assert index.ntotal == 2 and index.d == 2 and index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert load_index(tmp_path / "index", expected_model="embeddinggemma")[0].ntotal == 2
    with pytest.raises(ValueError, match="model differs"):
        load_index(tmp_path / "index", expected_model="different-model")
    client = FakeClient(embeddings=[[0, 5]])
    found = retrieve("backup?", index, metadata, top_k=9, client=client)
    assert [item["page"] for item in found] == [2, 1]
    assert found[0]["source"] == "demo.pdf" and found[0]["score"] == pytest.approx(1)
    with pytest.raises(ValueError, match="top_k"):
        retrieve("backup?", index, metadata, top_k=0, client=client)
    with pytest.raises(ValueError, match="dimension"):
        retrieve("backup?", index, metadata, client=FakeClient(embeddings=[[1, 2, 3]]))
    (tmp_path / "index" / "metadata.json").write_text("[]")
    with pytest.raises(ValueError, match="disagree"):
        load_index(tmp_path / "index")
    (tmp_path / "index" / "metadata.json").write_text(json.dumps(chunks))
    (tmp_path / "index" / "manifest.json").write_text(
        '{"embedding_model":"embeddinggemma","dimension":4}'
    )
    with pytest.raises(ValueError, match="manifest disagrees"):
        load_index(tmp_path / "index")


def test_prompt_answer_and_safe_references(tmp_path):
    chunks = chunks_fixture()
    build_index(np.asarray([[1, 0], [0, 1]], dtype=np.float32), chunks, tmp_path / "index")
    index, metadata = load_index(tmp_path / "index")
    client = FakeClient(embeddings=[[0, 2]])
    result = answer("When?", index, metadata, top_k=1, client=client)
    assert result["answer"] == "The backup is on Friday."
    assert result["sources"][0]["page"] == 2
    assert "text" not in result["sources"][0]
    assert set(result["timings"]) == {"retrieval_seconds", "generation_seconds"}
    messages = client.calls[1][1]["messages"]
    assert "untrusted" in SYSTEM_PROMPT and "insufficient" in SYSTEM_PROMPT
    assert "<untrusted_passage>" in messages[1]["content"]
    assert "Backup on Friday." in messages[1]["content"]
    with pytest.raises(ValueError, match="nonempty"):
        assemble_messages("", [])
    with pytest.raises(ValueError, match="empty answer"):
        answer("When?", index, metadata, client=FakeClient([[0, 2]], ""))


def test_demo_output_is_local(tmp_path):
    path = create_demo(tmp_path / "nested" / "demo.pdf")
    assert path == tmp_path / "nested" / "demo.pdf"
    assert Path(path).is_file()
