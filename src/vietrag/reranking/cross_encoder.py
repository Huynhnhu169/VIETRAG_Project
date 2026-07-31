"""Optional cross-encoder reranker with a deterministic lexical fallback."""

from __future__ import annotations

from vietrag.reranking.base import BaseReranker
from vietrag.retrieval.tokenization import tokenize
from vietrag.schemas import RetrievedPassage


def _rebuild(
    candidates: list[RetrievedPassage],
    scores: list[float],
    *,
    component_name: str,
    top_k: int,
) -> list[RetrievedPassage]:
    pairs = sorted(
        zip(candidates, scores),
        key=lambda item: (-float(item[1]), item[0].context_id),
    )[:top_k]
    output: list[RetrievedPassage] = []
    for rank, (candidate, score) in enumerate(pairs, start=1):
        components = dict(candidate.component_scores)
        components[component_name] = float(score)
        output.append(
            RetrievedPassage(
                context_id=candidate.context_id,
                document=candidate.document,
                rank=rank,
                score=float(score),
                component_scores=components,
            )
        )
    return output


class LexicalReranker(BaseReranker):
    """Offline fallback used for smoke tests, not a trained cross-encoder."""

    reranker_id = "lexical_overlap"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedPassage],
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        query_tokens = set(tokenize(query))
        scores: list[float] = []
        for candidate in candidates:
            title_tokens = set(tokenize(candidate.document.title))
            body_tokens = set(tokenize(candidate.document.raw_text))
            title_overlap = len(query_tokens & title_tokens) / max(
                len(query_tokens), 1
            )
            body_overlap = len(query_tokens & body_tokens) / max(
                len(query_tokens), 1
            )
            retrieval_tiebreak = 1e-6 / candidate.rank
            scores.append(0.35 * title_overlap + 0.65 * body_overlap + retrieval_tiebreak)
        return _rebuild(
            candidates,
            scores,
            component_name=self.reranker_id,
            top_k=min(top_k, len(candidates)),
        )


class CrossEncoderReranker(BaseReranker):
    """Lazy Sentence Transformers CrossEncoder backend."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 4,
    ) -> None:
        self.reranker_id = model_id
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "The cross-encoder is optional. Install requirements-ml.txt "
                    "or configure reranking.backend=lexical for offline execution."
                ) from exc
            device = None if self.device == "auto" else self.device
            self._model = CrossEncoder(
                self.model_id,
                revision=self.revision,
                device=device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedPassage],
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        pairs = [
            (query, f"{candidate.document.title}\n{candidate.document.raw_text}")
            for candidate in candidates
        ]
        scores = self._load().predict(
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size,
        ).tolist()
        return _rebuild(
            candidates,
            scores,
            component_name="cross_encoder",
            top_k=min(top_k, len(candidates)),
        )
