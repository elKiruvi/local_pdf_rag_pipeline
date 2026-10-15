# Pipeline architecture

## Quick path

`pdf-rag demo` creates an independently authored PDF. `ingest` converts extractable page text to `outputs/pages.json`; `chunk` writes `outputs/chunks.json`; `index` asks local Ollama for `embeddinggemma` vectors, normalizes them to L2 unit length, and writes `outputs/index/vectors.index`, `metadata.json`, and `manifest.json`. `retrieve` embeds a question with the same model and searches FAISS `IndexFlatIP`; for unit vectors, inner product ranks by cosine similarity. `ask` passes retrieved text as untrusted reference data to local `gemma3:4b` and returns answer, retrieved references, and separate timing fields. `evaluate` reports retrieval metrics for the newly authored example cases.

## Interfaces

| Module | Input / output | Boundary |
| --- | --- | --- |
| `ingest` | PDF path → nonempty `source`, one-based `page`, `text` records | Malformed/image-only PDFs fail with a helpful message. |
| `chunker` | Page records → numbered chunks with `section` and `content_type` | General heading and delimited-row heuristics; splits remain within a page. Character size/overlap are not tokens. |
| `embed` | Nonempty texts → contiguous, finite, normalized float32 matrix | Only a local Ollama client; checks count and dimension. |
| `build_index` | Unit vectors + aligned chunk records → FAISS and JSON | Rejects mismatched counts, non-unit vectors, invalid dimensions, and a wrong query embedding model on load. Files contain full private text and must stay local. |
| `retrieve` | Question + index/chunks → scored passages | Rejects incompatible query dimension and invalid top-k; each result retains source/page. |
| `rag` | Question + passages → answer, retrieved references, timings | The model may still hallucinate; references are not claim-level verified citations. |
| `evaluate_retrieval` | Labeled cases + search callable → macro averages | Precision@k divides by k (even if corpus is smaller), recall divides by number of relevant IDs, reciprocal rank uses first hit; empty-relevance recall is defined as zero. MRR is mean reciprocal rank across cases. |

The CLI defaults to ignored `outputs/` relative to the caller's working directory. Package imports never depend on that directory. The example corpus is built locally, not checked in. No network service other than a locally running Ollama server is intentionally used at runtime; set up Ollama to listen only on trusted interfaces. CI mocks its client and does not validate real-model answer quality, end-to-end accuracy, or large-corpus behavior. No automatic prompt/context/query logging occurs.
