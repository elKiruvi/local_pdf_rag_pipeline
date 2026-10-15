"""Small local command-line interface; no cloud fallback or raw-query logging."""

import argparse
import json
import sys
from pathlib import Path

from .build_index import build_index, load_index
from .chunker import chunk_pages
from .demo import create_demo
from .embed import EMBEDDING_MODEL, embed_texts
from .evaluate_retrieval import evaluate
from .ingest import extract_pages
from .rag import CHAT_MODEL, answer
from .retrieve import retrieve


def read_json(path: Path):
    if not path.is_file():
        raise ValueError("Required local input file is missing; run the preceding command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local PDF RAG pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate a synthetic example PDF")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo.pdf"))
    ingest = commands.add_parser("ingest", help="Extract PDF pages")
    ingest.add_argument("--pdf", required=True, type=Path)
    ingest.add_argument("--output", type=Path, default=Path("outputs/pages.json"))
    chunk = commands.add_parser("chunk", help="Split extracted pages")
    chunk.add_argument("--pages", type=Path, default=Path("outputs/pages.json"))
    chunk.add_argument("--output", type=Path, default=Path("outputs/chunks.json"))
    chunk.add_argument("--size", type=int, default=800)
    chunk.add_argument("--overlap", type=int, default=100)
    indexing = commands.add_parser("index", help="Embed and build a FAISS index")
    indexing.add_argument("--chunks", type=Path, default=Path("outputs/chunks.json"))
    indexing.add_argument("--output", type=Path, default=Path("outputs/index"))
    indexing.add_argument("--embedding-model", default=EMBEDDING_MODEL)
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
            pages = extract_pages(args.pdf)
            write_json(args.output, pages)
            print(f"Extracted {len(pages)} page(s) locally.")
        elif args.command == "chunk":
            chunks = chunk_pages(read_json(args.pages), args.size, args.overlap)
            write_json(args.output, chunks)
            print(f"Created {len(chunks)} local chunk(s).")
        elif args.command == "index":
            chunks = read_json(args.chunks)
            vectors = embed_texts([item["text"] for item in chunks], args.embedding_model)
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
