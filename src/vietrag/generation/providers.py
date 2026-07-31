"""Answer provider interfaces.

Providers receive selected evidence only. Citation objects are constructed by
the answerer from retrieved records, never accepted from provider text.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

from vietrag.chunking import split_sentences
from vietrag.retrieval.tokenization import tokenize
from vietrag.schemas import RetrievedPassage


class AnswerProvider(ABC):
    provider_id: str

    @abstractmethod
    def generate(self, query: str, evidence: list[RetrievedPassage]) -> str:
        raise NotImplementedError


class ExtractiveProvider(AnswerProvider):
    """Offline provider returning the best verbatim evidence sentence."""

    provider_id = "extractive"

    def generate(self, query: str, evidence: list[RetrievedPassage]) -> str:
        query_tokens = set(tokenize(query))
        candidates: list[tuple[float, int, str]] = []
        order = 0
        for passage in evidence:
            for sentence in split_sentences(passage.document.raw_text):
                sentence_tokens = set(tokenize(sentence))
                overlap = len(query_tokens & sentence_tokens) / max(
                    len(query_tokens), 1
                )
                candidates.append((overlap, -order, sentence))
                order += 1
        if not candidates:
            return ""
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


class OpenAICompatibleProvider(AnswerProvider):
    """Optional OpenAI-compatible chat-completions provider using stdlib HTTP."""

    provider_id = "openai_compatible"

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_base = (api_base or os.getenv("VIETRAG_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.getenv("VIETRAG_API_KEY", "")
        self.model = model or os.getenv("VIETRAG_API_MODEL", "")
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("VIETRAG_API_TIMEOUT_SECONDS", "60")
        )
        if not self.api_base or not self.api_key or not self.model:
            raise ValueError(
                "OpenAI-compatible provider requires VIETRAG_API_BASE, "
                "VIETRAG_API_KEY, and VIETRAG_API_MODEL."
            )

    def generate(self, query: str, evidence: list[RetrievedPassage]) -> str:
        evidence_text = "\n\n".join(
            f"[{item.context_id}] {item.document.title}\n{item.document.raw_text}"
            for item in evidence
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer in Vietnamese using only the supplied evidence. "
                        "Do not add citations or facts that are absent from it."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nEvidence:\n{evidence_text}",
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.timeout_seconds
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()
