# Security and privacy

Do not report vulnerabilities using a public issue if the report contains secrets, private documents, or reproductions derived from them. Use GitHub private vulnerability reporting if enabled; otherwise contact the maintainer privately without attaching confidential data. Never copy findings into PR comments or CI logs.

This local demonstration has no production security guarantee. It sends passages only to the loopback Ollama client, but local Ollama configuration, filesystem permissions, other processes and model behavior remain outside this package's control. Generated JSON and FAISS indexes contain private source text. **Load only trusted, locally generated indexes**: native FAISS parsing is not safe for arbitrary attacker-controlled files. Even a checksum alongside a mutable index detects corruption, not malicious replacement. `.gitignore` is not an access control.

| Threat | Relative risk | Boundary and residual risk |
| --- | --- | --- |
| Accidental publication of documents, derived data or credentials | High | Tracked-tree and reachable-history policy plus Gitleaks check common patterns; inspect every publication surface separately. |
| Supply-chain or CI compromise | High | Locked dependencies, pinned Actions, read-only token and audited releases reduce exposure, not eliminate it. |
| Malicious PDFs or injected passage instructions | Medium | Local extraction and system/data separation do not sandbox the PDF parser or guarantee model obedience. |
| Manipulated local FAISS index or metadata | Medium | Only trusted indexes should be loaded; native deserialization and writable local files are trust boundaries. |
| Oversized inputs or interrupted multi-file writes | Medium | Local resource limits and rollback checks are necessary; local disk failures remain possible. |
| Misleading retrieval-quality claims | Medium | Synthetic labels and offline mocks validate contracts, not real-world answer faithfulness. |

Before publication, run `python3 scripts/check_repository.py --history` and an independently installed Gitleaks `git --redact` scan. A locally stored, ignored pattern list can be passed using `--private-denylist <local-file>`; never commit or print the patterns. The script rejects unknown paths and oversized/non-text files, fails on incomplete scans, and reports only category counts. It inspects reachable local Git trees, **not** unreachable objects, forks or external clones. Git blobs are not GitHub metadata.

For a separate, **opt-in** public-surface audit, inspect PR bodies/reviews/comments, commit messages and metadata (approved public author attribution is allowed), tags/releases, Actions logs/artifacts, repository settings, and the exact SHA of every check. Use GitHub CLI/API with appropriate access; keep fetched material in ignored local storage, report aggregate status only and mark inaccessible surfaces *not verified*. A removed file does not erase public history or copies: stop publishing affected material, privately identify exposure surfaces and coordinate credential rotation with the holder. Do not rewrite public history automatically.
