# Experiment artifacts

Each P0–P5 configuration writes to its own directory:

- `predictions.jsonl`: query-level rankings, component scores, metrics, and latency.
- `metrics.json`: aggregate retrieval and robustness metrics.
- `metrics.csv`: one-row table for reporting.
- `run_metadata.json`: seed, dependency versions, hardware, model identifiers,
  revisions, split, and parameters.

Generated artifacts are Git-ignored by default. Review and deliberately publish
small result tables only after verifying dataset licensing and experiment
provenance. Model selection, fusion weights, top-k, reranking settings, and
abstention thresholds are validation-only decisions. A test run requires the
explicit `--frozen-config` guard.
