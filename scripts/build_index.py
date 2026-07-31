"""Build and persist an offline BM25 or hashing-dense index."""

from __future__ import annotations

import argparse
from pathlib import Path

from vietrag.retrieval import BM25Retriever, DenseRetriever, HashingEncoder
from vietrag.retrieval.index_store import save_index
from vietrag.schemas import DocumentRecord
from vietrag.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("indexes/default"))
    parser.add_argument("--backend", choices=("bm25", "hashing"), default="bm25")
    parser.add_argument("--dimensions", type=int, default=384)
    args = parser.parse_args()
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(args.corpus)]
    if args.backend == "bm25":
        retriever = BM25Retriever()
        configuration = {"backend": "bm25", "k1": 1.5, "b": 0.75}
    else:
        retriever = DenseRetriever(HashingEncoder(args.dimensions))
        configuration = {
            "backend": "hashing",
            "dimensions": args.dimensions,
            "warning": "offline fallback; not a trained semantic encoder",
        }
    retriever.index(documents)
    save_index(args.output, retriever, configuration=configuration)
    print(f"Saved {retriever.system_id} index and separate corpus metadata to {args.output}")


if __name__ == "__main__":
    main()
