"""Run the same tested pipeline used by Streamlit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vietrag.config import load_config
from vietrag.pipeline import RAGPipeline
from vietrag.schemas import DocumentRecord
from vietrag.utils.io import read_jsonl


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid_rrf", "hybrid_alpha"),
        default=None,
    )
    parser.add_argument("--ambiguous", action="store_true")
    args = parser.parse_args()
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(args.corpus)]
    pipeline = RAGPipeline(
        documents, load_config(args.config), retrieval_mode=args.mode
    )
    result = pipeline.ask(args.query, ambiguous=args.ambiguous)
    print(json.dumps(result.prediction.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
