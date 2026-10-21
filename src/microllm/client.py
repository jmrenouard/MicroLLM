"""Typed Python SDK for MicroLLM (`POST /v1/systemone`), compatible with TypeSafe & poorjev idioms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union
import httpx


@dataclass(frozen=True)
class Noul:
    """Typed helper for a Yes/No (`noul`) question."""

    instructions: str | None = None
    true_description: str | None = None
    false_description: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul"}
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        criteria: dict[str, str] = {}
        if self.true_description is not None:
            criteria["true"] = self.true_description
        if self.false_description is not None:
            criteria["false"] = self.false_description
        if criteria:
            payload["criteria"] = criteria
        return payload


@dataclass(frozen=True)
class Choice:
    """Typed helper for a mutually-exclusive `choice` question (2..255 options)."""

    options: Union[Sequence[str], Mapping[str, str | None]]
    instructions: str = "Select the most appropriate option."

    def to_payload(self) -> dict[str, Any]:
        if isinstance(self.options, Mapping):
            criteria = dict(self.options)
        else:
            criteria = {str(opt): None for opt in self.options}
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": criteria,
        }


@dataclass(frozen=True)
class Score:
    """Typed helper for an ordinal `score` rubric question (2..10 levels)."""

    levels: Sequence[Any]
    instructions: str = "Rate on the given ordered scale."

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }


@dataclass(frozen=True)
class Multi:
    """Typed helper for an independent multi-label `multi` question (2..255 labels)."""

    options: Union[Sequence[str], Mapping[str, str | None]]
    instructions: str = "Select all applicable labels."
    threshold: float = 0.5

    def to_payload(self) -> dict[str, Any]:
        if isinstance(self.options, Mapping):
            criteria = dict(self.options)
        else:
            criteria = {str(opt): None for opt in self.options}
        return {
            "type": "multi",
            "instructions": self.instructions,
            "criteria": criteria,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class NoulResult:
    """Parsed answer for a `noul` question."""

    type: str
    probability: float
    threshold: float = 0.5

    @property
    def value(self) -> bool:
        return self.probability >= self.threshold


@dataclass(frozen=True)
class ChoiceResult:
    """Parsed answer for a `choice` question."""

    type: str
    value: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreResult:
    """Parsed answer for a `score` question."""

    type: str
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float

    @property
    def value(self) -> Any:
        best_idx = max(self.probabilities, key=lambda k: self.probabilities[k])
        return self.legend[best_idx]


@dataclass(frozen=True)
class MultiResult:
    """Parsed answer for an independent multi-label `multi` question."""

    type: str
    labels: list[str]
    probabilities: dict[str, float]
    confidence: float

    @property
    def value(self) -> list[str]:
        return self.labels


ParsedResult = Union[NoulResult, ChoiceResult, ScoreResult, MultiResult]


class MicroJevClient:
    """Typed System-One client for local or remote `/v1/systemone` endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "local-jev-key",
        model: str = "jev-latest",
        http_client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Union[Noul, Choice, Score, Multi, dict[str, Any]]],
        model: str | None = None,
    ) -> dict[str, ParsedResult]:
        """Send a batched System-One request and return strongly-typed answer objects."""
        serialized_questions: dict[str, Any] = {}
        for qid, q_spec in questions.items():
            if hasattr(q_spec, "to_payload"):
                serialized_questions[qid] = q_spec.to_payload()  # type: ignore[union-attr]
            else:
                serialized_questions[qid] = dict(q_spec)  # type: ignore[arg-type]

        payload = {
            "model": model or self.model,
            "state": state,
            "questions": serialized_questions,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        if self._http is not None:
            resp = self._http.post("/v1/systemone", json=payload, headers=headers)
        else:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                resp = client.post("/v1/systemone", json=payload, headers=headers)

        resp.raise_for_status()
        body = resp.json()
        raw_answers = body["answers"]

        parsed: dict[str, ParsedResult] = {}
        for qid, ans in raw_answers.items():
            q_type = ans["type"]
            if q_type == "noul":
                parsed[qid] = NoulResult(type="noul", probability=float(ans["noul"]))
            elif q_type == "choice":
                parsed[qid] = ChoiceResult(
                    type="choice",
                    value=str(ans["choice"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "score":
                parsed[qid] = ScoreResult(
                    type="score",
                    score=float(ans["score"]),
                    legend=dict(ans["legend"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "multi":
                parsed[qid] = MultiResult(
                    type="multi",
                    labels=list(ans["labels"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
        return parsed
