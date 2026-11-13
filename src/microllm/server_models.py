"""Pydantic request and response schemas for MicroLLM HTTP server endpoints."""

from __future__ import annotations

from typing import Any
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from microllm.schemas import (
    QuestionSpec,
    StructuredText,
    _ensure_finite_json,
)


class StudioAnalyzeRequest(BaseModel):
    """Request schema for Decision Studio side-by-side Gear 1 / Gear 2 / Cascade analysis."""

    state: StructuredText
    questions: dict[str, QuestionSpec]
    gate_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    run_order_flip_test: bool = True

    @field_validator("state", mode="before")
    @classmethod
    def validate_studio_state(cls, v: Any) -> StructuredText:
        if v is None or not isinstance(v, (str, dict, list)):
            raise ValueError("state must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)

    @model_validator(mode="after")
    def validate_studio_questions(self) -> "StudioAnalyzeRequest":
        if not self.questions:
            raise ValueError("questions must be a non-empty object.")
        if not np.isfinite(self.gate_threshold) or not np.isfinite(self.alpha):
            raise ValueError("gate_threshold and alpha must be finite numbers.")
        return self


def _reject_bools_recursively(val: Any, path: str = "value") -> None:
    """Recursively check for boolean values and raise ValueError if detected."""
    if isinstance(val, bool):
        raise ValueError(f"{path} cannot be a boolean.")
    if isinstance(val, (list, tuple)):
        for i, item in enumerate(val):
            _reject_bools_recursively(item, f"{path}[{i}]")


class StudioRLCDDemoRequest(BaseModel):
    """Request schema for running a live local RLCD Stage 1 + Stage 2 training & calibration run."""

    stage1_steps: int = Field(default=35, ge=5, le=100)
    stage2_steps: int = Field(default=25, ge=5, le=100)
    use_rlcd2_weights: bool = True
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    features: list[list[float]] | None = None
    labels: list[int] | None = None

    @field_validator("features", "labels", mode="before")
    @classmethod
    def validate_no_bools_rlcd(cls, v: Any, info: Any) -> Any:
        if v is None:
            return None
        _reject_bools_recursively(v, info.field_name)
        return v

    @model_validator(mode="after")
    def validate_rlcd_arrays(self) -> "StudioRLCDDemoRequest":
        if self.features is None and self.labels is None:
            return self
        if self.features is None or self.labels is None:
            raise ValueError("features and labels must both be provided, or both omitted.")
        if not self.features or not self.labels:
            raise ValueError("features and labels must be non-empty lists.")
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Sample count mismatch: features has {len(self.features)} samples, labels has {len(self.labels)} items."
            )
        if len(self.features) > 10000:
            raise ValueError(
                f"Sample count N={len(self.features)} exceeds maximum limit of 10,000."
            )
        _ensure_finite_json(self.features)
        first_row = self.features[0]
        if not isinstance(first_row, list) or len(first_row) < 1:
            raise ValueError("features must be a 2D array of numeric values with dimension >= 1.")
        d_feat = len(first_row)
        if d_feat > 1024:
            raise ValueError(
                f"Feature dimension d_feat={d_feat} exceeds maximum limit of 1,024."
            )
        for i, row in enumerate(self.features):
            if not isinstance(row, list) or len(row) != d_feat:
                raise ValueError(
                    f"features row {i} has length {len(row) if isinstance(row, list) else 'non-list'}, expected {d_feat}."
                )
        for i, y in enumerate(self.labels):
            if isinstance(y, bool) or not isinstance(y, int) or y < 0:
                raise ValueError(f"labels[{i}] must be a non-negative integer, got {y}.")
            if y >= 256:
                raise ValueError(f"labels[{i}]={y} exceeds maximum allowed class index 255 (K <= 256).")
        k_classes = max(self.labels) + 1
        if k_classes < 2:
            raise ValueError(f"labels must contain at least 2 distinct classes (max label was {max(self.labels)}).")
        if k_classes > 256:
            raise ValueError(f"Number of classes K={k_classes} exceeds maximum limit of 256.")
        return self


class CalibrateRequest(BaseModel):
    """Request schema for `POST /v1/calibrate` (Per-K temperature scaling + Split Conformal + Clopper-Pearson)."""

    logits: list[list[float]] | None = None
    labels: list[int] | None = None
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    target_risk: float = Field(default=0.10, gt=0.0, lt=1.0)
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("logits", "labels", mode="before")
    @classmethod
    def validate_no_bools_calibrate(cls, v: Any, info: Any) -> Any:
        if v is None:
            return None
        _reject_bools_recursively(v, info.field_name)
        return v

    @model_validator(mode="after")
    def validate_calibration_arrays(self) -> "CalibrateRequest":
        if self.logits is None and self.labels is None:
            return self
        if not self.logits or not self.labels or len(self.logits) != len(self.labels):
            raise ValueError("logits and labels must be non-empty lists of equal length.")
        if len(self.logits) < 2:
            raise ValueError("logits and labels must contain at least 2 samples.")
        if len(self.logits) > 10000:
            raise ValueError(
                f"Sample count N={len(self.logits)} exceeds maximum limit of 10,000."
            )
        _ensure_finite_json(self.logits)
        first_row = self.logits[0]
        if not isinstance(first_row, list) or len(first_row) < 2:
            raise ValueError("Each logit vector must have at least 2 classes.")
        k = len(first_row)
        if k > 256:
            raise ValueError(f"Number of classes K={k} exceeds maximum limit of 256.")
        for i, row in enumerate(self.logits):
            if not isinstance(row, list) or len(row) != k:
                raise ValueError(
                    f"All logit vectors must have identical cardinality K={k}, got row {i} length {len(row) if isinstance(row, list) else 'non-list'}."
                )
        for i, y in enumerate(self.labels):
            if isinstance(y, bool) or not isinstance(y, int):
                raise ValueError(f"labels[{i}] must be an integer, got {type(y).__name__}.")
            if y < 0 or y >= k:
                raise ValueError(f"Each label must be in 0..{k - 1}, got labels[{i}]={y}.")
        return self


class GuardrailRequest(BaseModel):
    """Request schema for `POST /v1/guardrail` (Zero-Shot AI Safety Guardrail pack, arXiv:2609.29429)."""

    state: StructuredText
    context: StructuredText | None = None
    threshold: float = Field(default=0.55, ge=0.1, le=0.99)
    custom_rubrics: dict[str, str] | None = None

    @field_validator("state", mode="before")
    @classmethod
    def validate_guardrail_state(cls, v: Any) -> StructuredText:
        if v is None or not isinstance(v, (str, dict, list)):
            raise ValueError("state must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)

    @field_validator("context", mode="before")
    @classmethod
    def validate_guardrail_context(cls, v: Any) -> StructuredText | None:
        if v is None:
            return None
        if not isinstance(v, (str, dict, list)):
            raise ValueError("context must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)


class ProfilePullRequest(BaseModel):
    """Request schema for `POST /v1/profiles/pull` (`microllm pull <profile>`)."""

    profile: str
    download_weights: bool = False
    from_hub: bool = False

    @field_validator("profile")
    @classmethod
    def validate_profile_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("profile name must be a non-empty string.")
        return v.strip()


class BatchRequest(BaseModel):
    """Request schema for `POST /v1/batch` (Bulk CSV/JSONL decision evaluation)."""

    items: list[StructuredText]
    questions: dict[str, QuestionSpec]
    model: str = "microllm-gear1-fast"
    multilingual_strategy: str | None = None
    neural_model: str | None = None

    @model_validator(mode="after")
    def validate_batch_payload(self) -> "BatchRequest":
        if self.items is None or len(self.items) > 500:
            raise ValueError("items must contain at most 500 states.")
        _ensure_finite_json(self.items)
        if self.questions is None:
            raise ValueError("questions must not be None.")
        return self


__all__ = [
    "StudioAnalyzeRequest",
    "StudioRLCDDemoRequest",
    "CalibrateRequest",
    "GuardrailRequest",
    "ProfilePullRequest",
    "BatchRequest",
    "_ensure_finite_json",
    "_reject_bools_recursively",
]
