"""Small local command-line interface; no cloud fallback or raw-query logging."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .build_index import build_index, load_index
from .chunker import chunk_pages
from .demo import create_demo
from .embed import EMBEDDING_MODEL, chunk_embedding_text, embed_texts
from .evaluate_retrieval import evaluate
from .ingest import extract_pages
from .rag import CHAT_MODEL, answer
from .retrieve import retrieve


def read_json(path: Path):
    if not path.is_file():
        raise ValueError("Required local input file is missing; run the preceding command first.")
    if path.is_symlink() or path.stat().st_size > 50_000_000:
        raise ValueError("Local JSON is a symlink or exceeds the file size limit.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("JSON output must not replace a symlink.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local PDF RAG pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate a synthetic example PDF")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo.pdf"))
    ingest = commands.add_parser("ingest", help="Extract PDF pages")
    ingest.add_argument("--pdf", required=True, type=Path)
    ingest.add_argument("--output", type=Path, default=Path("outputs/pages.json"))
    ingest.add_argument("--max-pdf-mb", type=int, default=100)
    ingest.add_argument("--max-pages", type=int, default=1_000)
    ingest.add_argument("--max-text-chars", type=int, default=10_000_000)
    chunk = commands.add_parser("chunk", help="Split extracted pages")
    chunk.add_argument("--pages", type=Path, default=Path("outputs/pages.json"))
    chunk.add_argument("--output", type=Path, default=Path("outputs/chunks.json"))
    chunk.add_argument("--size", type=int, default=800)
    chunk.add_argument("--overlap", type=int, default=100)
    chunk.add_argument("--max-chunks", type=int, default=100_000)
    indexing = commands.add_parser("index", help="Embed and build a FAISS index")
    indexing.add_argument("--chunks", type=Path, default=Path("outputs/chunks.json"))
    indexing.add_argument("--output", type=Path, default=Path("outputs/index"))
    indexing.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    indexing.add_argument("--batch-size", type=int, default=32)
    for name in ("retrieve", "ask"):
        cmd = commands.add_parser(
            name, help="Search local index" if name == "retrieve" else "Ask local model"
        )
        cmd.add_argument("question")
        cmd.add_argument("--index", type=Path, default=Path("outputs/index"))
        cmd.add_argument("--top-k", type=int, default=3)
        cmd.add_argument("--embedding-model", default=EMBEDDING_MODEL)
        if name == "ask":
            cmd.add_argument("--chat-model", default=CHAT_MODEL)
    metrics = commands.add_parser("evaluate", help="Evaluate labeled retrieval cases")
    metrics.add_argument("--cases", type=Path, default=Path("examples/evaluation.json"))
    metrics.add_argument("--index", type=Path, default=Path("outputs/index"))
    metrics.add_argument("--top-k", type=int, default=3)
    metrics.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            create_demo(args.output)
            print("Synthetic PDF created locally.")
        elif args.command == "ingest":
            pages = extract_pages(
                args.pdf, args.max_pdf_mb * 1_000_000, args.max_pages, args.max_text_chars
            )
            write_json(args.output, pages)
            print(f"Extracted {len(pages)} page(s) locally.")
        elif args.command == "chunk":
            chunks = chunk_pages(read_json(args.pages), args.size, args.overlap, args.max_chunks)
            write_json(args.output, chunks)
            print(f"Created {len(chunks)} local chunk(s).")
        elif args.command == "index":
            chunks = read_json(args.chunks)
            vectors = embed_texts(
                [chunk_embedding_text(item) for item in chunks],
                args.embedding_model,
                batch_size=args.batch_size,
            )
            build_index(vectors, chunks, args.output, args.embedding_model)
            print(f"Indexed {len(chunks)} local chunk(s).")
        else:
            index, chunks = load_index(args.index, args.embedding_model)
            if args.command == "evaluate":
                cases = read_json(args.cases)
                result = evaluate(
                    cases,
                    lambda q, k: retrieve(q, index, chunks, k, args.embedding_model),
                    args.top_k,
                )
            elif args.command == "retrieve":
                result = retrieve(args.question, index, chunks, args.top_k, args.embedding_model)
            else:
                result = answer(
                    args.question, index, chunks, args.top_k, args.embedding_model, args.chat_model
                )
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        # Avoid leaking local paths, document text, or raw prompts in error output.
        print(
            f"Error: {type(exc).__name__}. Check local inputs and model availability.",
            file=sys.stderr,
        )
        return 1
    return 0
