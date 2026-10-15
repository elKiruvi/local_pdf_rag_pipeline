# Pipeline architecture

## Quick path

`pdf-rag demo` creates an independently authored PDF. `ingest` converts extractable page text to `outputs/pages.json`; `chunk` writes `outputs/chunks.json`; `index` embeds section heading plus chunk text in bounded batches using local Ollama `embeddinggemma`, normalizes vectors to L2 unit length, and writes `outputs/index/vectors.index`, `metadata.json`, and `manifest.json`. Original chunk text remains unchanged for answers and references. Version 2 indexes require a rebuild from earlier versions. `retrieve` embeds a question with the same model and searches FAISS `IndexFlatIP`; for unit vectors, inner product ranks by cosine similarity. `ask` passes retrieved text as untrusted reference data to local `gemma3:4b` and returns answer, retrieved references, and separate timing fields. `evaluate` reports retrieval metrics for the newly authored example cases.

## Interfaces

| Module | Input / output | Boundary |
| --- | --- | --- |
| `ingest` | PDF path → nonempty `source`, one-based `page`, `text` records | Malformed/image-only PDFs fail with a helpful message. |
| `chunker` | Page records → numbered chunks with `section` and `content_type` | General heading and pipe/tab-delimited-row heuristics after cleaning; splits remain within a page. This is not PDF table extraction. Character size/overlap are not tokens. |
| `embed` | Nonempty texts → contiguous, finite, normalized float32 matrix | Only a local Ollama client; bounded batches preserve order and check count and dimension across responses. |
| `build_index` | Unit vectors + aligned chunk records → FAISS and JSON | Validates schema, types, sizes, unique IDs, checksums and model before native loading; old manifests require a rebuild. Files contain full private text. Same-directory hashes detect corruption, not hostile replacements. Load only trusted local indexes. |
| `retrieve` | Question + index/chunks → scored passages | Rejects incompatible query dimension and invalid top-k; each result retains source/page. |
| `rag` | Question + passages → answer, retrieved references, timings | The model may still hallucinate; references are not claim-level verified citations. |
| `evaluate_retrieval` | Labeled cases + search callable → macro averages | Precision@k divides by k (even if corpus is smaller), recall divides by number of relevant IDs, reciprocal rank uses first hit; empty-relevance recall is defined as zero. MRR is mean reciprocal rank across cases. |

The CLI defaults to ignored `outputs/` relative to the caller's working directory. Package imports never depend on that directory. The example corpus is built locally, not checked in. Requests are pinned to loopback Ollama (`127.0.0.1:11434`) with proxy environment handling disabled; there is no cloud fallback or remote-host option. Set up Ollama to listen only on trusted interfaces. CI mocks its client and does not validate real-model answer quality, end-to-end accuracy, or large-corpus behavior. No automatic prompt/context/query logging occurs.

Index creation validates a staged sibling directory before replacing a prior valid index. A failed normal replacement restores the backup; directory renames are **not** universally atomic or crash-proof, and a failed rollback may require manual recovery from the sibling backup. JSON CLI output uses a temporary sibling file and `os.replace`. The defaults limit PDF input to 100 MB, 1,000 pages, 10 million decoded characters, 100,000 chunks and 50 MB per JSON input; `ingest` and `chunk` expose CLI options for smaller limits. Vectors, metadata and index files have additional fixed upper bounds. Parser memory use for a single hostile PDF page is not fully bounded by these checks.
