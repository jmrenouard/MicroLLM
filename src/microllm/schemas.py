"""Strict Pydantic v2 schemas for the Jev-compatible `/v1/systemone` API contract (32/32 MUST)."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StructuredText = Union[str, dict[str, Any], list[Any]]


def _ensure_finite_json(value: Any) -> Any:
    """Recursively verify that a JSON value contains no NaN or Infinity floats (`jevcompat` §4.1/§6)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers (NaN or Infinity) are not valid JSON.")
    if isinstance(value, dict):
        for k, v in value.items():
            _ensure_finite_json(k)
            _ensure_finite_json(v)
    elif isinstance(value, list):
        for item in value:
            _ensure_finite_json(item)
    return value


def render_structured_text(value: StructuredText | None, fallback: str = "") -> str:
    """Convert string, JSON object, or JSON array into a deterministic text representation."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class NoulQuestion(BaseModel):
    """Yes/No probabilistic proposition (`type: 'noul'`)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None] | None = None

    @field_validator("instructions")
    @classmethod
    def validate_instructions_finite(cls, v: StructuredText | None) -> StructuredText | None:
        return _ensure_finite_json(v)

    @field_validator("criteria")
    @classmethod
    def validate_noul_criteria(
        cls, v: dict[str, StructuredText | None] | None
    ) -> dict[str, StructuredText | None] | None:
        if v is None:
            return None
        _ensure_finite_json(v)
        allowed = {"true", "false"}
        invalid = set(v.keys()) - allowed
        if invalid:
            raise ValueError(f"Noul criteria keys must be subset of {allowed}, got {invalid}")
        return v


class ChoiceQuestion(BaseModel):
    """Mutually-exclusive discrete selection among 2..255 options (`type: 'choice'`)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None]

    @field_validator("instructions")
    @classmethod
    def validate_instructions_finite(cls, v: StructuredText | None) -> StructuredText | None:
        return _ensure_finite_json(v)

    @field_validator("criteria")
    @classmethod
    def validate_choice_options(
        cls, v: dict[str, StructuredText | None]
    ) -> dict[str, StructuredText | None]:
        if not isinstance(v, dict):
            raise ValueError("Choice criteria must be a mapping of option names to descriptions.")
        _ensure_finite_json(v)
        n = len(v)
        if n < 2 or n > 255:
            raise ValueError(f"Choice criteria must contain between 2 and 255 options (got {n}).")
        for k in v:
            if not isinstance(k, str) or len(k) == 0:
                raise ValueError("Choice option names must be non-empty strings.")
        return v


class ScoreQuestion(BaseModel):
    """Ordinal rubric scale evaluation across 2..10 levels (`type: 'score'`)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    instructions: StructuredText | None = None
    criteria: list[StructuredText]

    @field_validator("instructions")
    @classmethod
    def validate_instructions_finite(cls, v: StructuredText | None) -> StructuredText | None:
        return _ensure_finite_json(v)

    @field_validator("criteria", mode="before")
    @classmethod
    def reject_map_and_bounds(cls, v: Any) -> list[StructuredText]:
        if not isinstance(v, list):
            raise ValueError("Score criteria must be an ordered JSON array (list) of levels.")
        _ensure_finite_json(v)
        n = len(v)
        if n < 2 or n > 10:
            raise ValueError(f"Score criteria must have between 2 and 10 levels (got {n}).")
        for item in v:
            if item is None or not isinstance(item, (str, dict, list)):
                raise ValueError("Each score level description must be a string, object, or array.")
        return v


class MultiQuestion(BaseModel):
    """Multi-label independent classification across 2..255 labels (`type: 'multi'`)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["multi"] = "multi"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("instructions")
    @classmethod
    def validate_instructions_finite(cls, v: StructuredText | None) -> StructuredText | None:
        return _ensure_finite_json(v)

    @field_validator("criteria")
    @classmethod
    def validate_multi_options(
        cls, v: dict[str, StructuredText | None]
    ) -> dict[str, StructuredText | None]:
        if not isinstance(v, dict):
            raise ValueError("Multi criteria must be a mapping of label names to descriptions.")
        _ensure_finite_json(v)
        n = len(v)
        if n < 2 or n > 255:
            raise ValueError(f"Multi criteria must contain between 2 and 255 labels (got {n}).")
        for k in v:
            if not isinstance(k, str) or len(k) == 0:
                raise ValueError("Multi label names must be non-empty strings.")
        return v


QuestionSpec = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, MultiQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """Top-level request body for `POST /v1/systemone`."""

    model_config = ConfigDict(extra="allow")

    model: str = "jev-latest"
    state: StructuredText
    questions: dict[str, QuestionSpec]

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, v: Any) -> StructuredText:
        if v is None or not isinstance(v, (str, dict, list)):
            raise ValueError("state must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)

    @model_validator(mode="after")
    def validate_questions_non_empty(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("questions must be a non-empty object.")
        for qid in self.questions:
            if not isinstance(qid, str) or len(qid) == 0:
                raise ValueError("Every question id must be a non-empty string.")
        return self



class NoulAnswer(BaseModel):
    """Spec-exact Noul response (`jevcompat` §4.2): no unprefixed `confidence` field."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    """Spec-exact Choice response (`jevcompat` §4.3)."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    """Spec-exact Score response (`jevcompat` §4.4)."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, StructuredText]
    probabilities: dict[str, float]
    confidence: float


class MultiAnswer(BaseModel):
    """Multi-label independent classification response (`type: 'multi'`)."""

    type: Literal["multi"] = "multi"
    labels: list[str]
    probabilities: dict[str, float]
    confidence: float


AnswerSpec = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, MultiAnswer]


class UsageInfo(BaseModel):
    """Token accounting (`jevcompat` §4.1: non-negative JSON integers)."""

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    """Top-level response envelope for `POST /v1/systemone`."""

    model: str
    answers: dict[str, AnswerSpec]
    usage: UsageInfo
