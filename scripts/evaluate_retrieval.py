"""Evaluate P0-P4 and write query-level, aggregate, and run artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from vietrag.evaluation.runner import evaluate_retriever, save_evaluation_artifacts
from vietrag.retrieval.factory import build_reranker
from vietrag.config import load_config
from vietrag.retrieval.factory import build_retriever
from vietrag.schemas import DocumentRecord, QueryRecord
from vietrag.utils.io import read_jsonl
from vietrag.utils.reproducibility import collect_run_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/processed/queries.with_splits.jsonl"),
    )
    parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid_rrf", "hybrid_alpha"),
        default=None,
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--rerank", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        help="Required for a test-set run; proves selection was frozen first.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    split = args.split or config.get("evaluation", {}).get("split", "validation")
    mode = args.mode or config["retrieval"]["mode"]
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    output = args.output or Path(config["paths"]["experiment_dir"])
    if split == "test" and (
        args.frozen_config is None or not args.frozen_config.is_file()
    ):
        raise SystemExit(
            "Test evaluation requires --frozen-config. Select models and "
            "thresholds on validation first."
        )
    set_seed(seed)
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(args.corpus)]
    all_queries = [QueryRecord.from_dict(row) for row in read_jsonl(args.queries)]
    queries = [query for query in all_queries if query.split == split]
    if not queries:
        raise SystemExit(f"No queries found for split={split}")
    retriever = build_retriever(documents, config, mode=mode)
    reranker_config = config
    if args.rerank and not config.get("reranking", {}).get("enabled", False):
        reranker_config = deepcopy(config)
        reranker_config.setdefault("reranking", {})["enabled"] = True
    reranker = (
        build_reranker(reranker_config)
        if args.rerank or config.get("reranking", {}).get("enabled", False)
        else None
    )
    rows, summary = evaluate_retriever(
        retriever,
        queries,
        top_k=10,
        reranker=reranker,
        reranker_input_k=20,
        reranker_output_k=5,
    )
    metadata = collect_run_metadata(seed)
    metadata.update(
        {
            "dataset_sources": sorted({document.source for document in documents}),
            "split": split,
            "retrieval_mode": mode,
            "reranker": reranker.reranker_id if reranker else None,
            "model_identifiers": {
                "dense": config["retrieval"]["dense"]["model_id"]
                if config["retrieval"]["dense"]["backend"] == "sentence_transformers"
                else "hashing",
                "dense_revision": config["retrieval"]["dense"].get("model_revision"),
                "reranker": reranker.reranker_id if reranker else None,
                "reranker_revision": None,
            },
            "parameters": {
                "top_k": config["retrieval"]["top_k"],
                "candidate_k": config["retrieval"]["candidate_k"],
                "rrf_k": config["retrieval"]["fusion"]["rrf_k"],
                "alpha": config["retrieval"]["fusion"]["alpha"],
                "abstention_threshold": config["generation"]["abstention_threshold"],
            },
            "configuration_file": str(args.config),
        }
    )
    save_evaluation_artifacts(output, rows, summary, metadata)
    print(f"Evaluated {len(queries)} {split} queries; artifacts: {output}")


if __name__ == "__main__":
    main()
