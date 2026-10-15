# Future improvements (not implemented)

This project intentionally remains a small local demonstration. Every item below is **not currently implemented**.

| Priority | Candidate | Why it is deferred |
| --- | --- | --- |
| High | Verified faithfulness and claim-level citations | Retrieved references alone cannot prove each generated claim. |
| High | OCR and native PDF table extraction | Image-only pages and complex table layouts are not supported. |
| High | Configurable, token-aware chunking | Current character splitting and simple delimiter heuristic can lose layout or split context. |
| Medium | Hybrid lexical/semantic search and reranking | Would add dependencies and require representative offline benchmarks. |
| Medium | Incremental indexing and versioned corpora | Current index is rebuilt for a small local corpus. |
| Medium | Larger real-model retrieval benchmarks and multi-hardware profiling | Require pulled models, documented model digests, hardware and explicit reproducible inputs. |
| Medium | Advanced adversarial prompt-injection evaluation | Mocked message-construction tests cannot establish real-model robustness. |
| Low | Opt-in sanitized observability | Must avoid raw-query/document logging and remain local by default. |
| Low | Optional API/UI, Docker, and multi-user operations | Out of scope until a use case justifies a service and threat model. |
| Low | Release/deployment automation | No hosted service, registry release or established license grant exists to deploy. |
