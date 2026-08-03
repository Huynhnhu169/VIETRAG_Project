# ViRHE4QA validation results

These are measured Direction 1 results, not estimates. They were produced on
Google Colab with 1,544 clean queries from the frozen context-disjoint
validation split. The source artifact archive has SHA-256
`CA1FC277CDD2DE6BB2A867FB72FB3515D7C3B8062387F825260B91063CE9BEE8`.

| Pipeline | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| P0 BM25 | 0.7085 | 0.9229 | 0.9624 | 0.8036 | 0.8427 | 2.22 ms |
| P1 BGE-M3 | 0.5920 | 0.8329 | 0.8983 | 0.6956 | 0.7448 | 41.40 ms |
| P2 hybrid RRF | 0.6775 | 0.9113 | 0.9579 | 0.7765 | 0.8208 | 44.80 ms |
| P4 hybrid + BGE reranker | **0.8497** | **0.9741** | n/a | **0.9036** | **0.9216** | 2,623.97 ms |

The archived P4 run reranked 20 candidates but retained only five. Its emitted
`recall@10` therefore duplicated Recall@5 and is intentionally reported as
`n/a` here. The evaluator has since been corrected to retain enough reranked
results for every requested cutoff; P4 must be rerun before publishing a true
Recall@10.

## Findings

- BM25 is the strongest non-reranked baseline and the practical fast mode.
- Dense-only BGE-M3 underperforms BM25 on this clause-level legal corpus.
- Equal-weight RRF underperforms BM25 at early ranks, so fusion weights or a
  lexical-heavy alpha should be selected on validation.
- The cross-encoder produces the best ranking quality but raises mean latency
  to 2.62 seconds per query.
- P4 improves Recall@1 over P0 by 0.1412 (paired bootstrap 95% CI
  [0.1185, 0.1632]) and MRR@10 by 0.1000 (95% CI [0.0857, 0.1143]).
- P4 corrects 273 P0 top-1 failures and loses 55 P0 top-1 successes, a net
  improvement of 218 queries.

## Remaining work

- Run P3 lexical-heavy alpha fusion on validation.
- Rerun corrected P4 and compare reranker input sizes 10 and 20.
- Evaluate the frozen robustness subset by noise type.
- Tune citation/abstention behavior and evaluate P5 generation separately.
- Freeze the selected configuration before the single guarded test run.
- Record PyTorch, CUDA, GPU, VRAM, and immutable model revisions in metadata.
