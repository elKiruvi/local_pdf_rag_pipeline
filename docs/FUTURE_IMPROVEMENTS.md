# Future improvements (not implemented)

This project intentionally remains a small local demonstration. Every item below is **not currently implemented**.

| Priority | Candidate | Why it is deferred |
| --- | --- | --- |
| High | Faithfulness and claim-level citation evaluation | Retrieved references alone cannot prove each generated claim. |
| High | Better table extraction and configurable chunking | Current character splitting and simple delimiter heuristic can lose layout. |
| Medium | Hybrid lexical/semantic search and reranking | Would add dependencies and require representative offline benchmarks. |
| Medium | Incremental indexing and versioned corpora | Current index is rebuilt for a small local corpus. |
| Medium | Local integration benchmarks and large-corpus profiling | Require pulled models, hardware, and explicit reproducible inputs. |
| Low | Richer local observability | Should remain opt-in and avoid raw-query logging by default. |
| Low | Optional API/UI, Docker, and multi-user operations | Out of scope until a use case justifies a service and threat model. |
| Low | Release/CD automation | No hosted service or registry release exists to deploy. |
