# Data

No upstream dataset is distributed in this repository.

## Licenses and permitted use

- **ViRHE4QA** is the primary corpus. Its source describes research-only,
  non-commercial use. Re-check the current upstream terms before every public
  release or commercial use.
- **EduCoQA** is reserved for cleaned natural-query/OOD evaluation and is
  described by ViRE as CC BY-NC 4.0.
- **ViLexNorm** (CC BY-NC-SA 4.0), **ViSP**, **UIT-ViQuAD 2.0**, and
  **Viwiki-Spelling** are not redistributed. Consult their upstream licenses.

`samples/raw_synthetic.jsonl` was authored for this project. It is small,
synthetic, and intended only for tests and smoke runs.

## Required order of operations

1. Download data locally after accepting the upstream terms.
2. Parse and canonicalize contexts.
3. Compute deterministic context hashes and deduplicate.
4. Freeze context-disjoint and document-disjoint manifests.
5. Verify leakage checks.
6. Only then generate noisy query variants.

The original ViRHE4QA split is retained only as provenance and is never used to
claim retrieval performance.

## Local directories

- `raw/`: downloaded archives (ignored by Git).
- `interim/`: parsed temporary artifacts (ignored by Git).
- `processed/`: canonical corpus and query files (ignored by Git).
- `manifests/`: generated, frozen split manifests (ignored by Git).
- `samples/`: project-authored synthetic fixtures (tracked).
