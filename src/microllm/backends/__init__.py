"""Decision engine backends: Gear 1 Fast Encoder, Gear 2 AnyJev Prefill Readout, and StartLux Cascade."""

from microllm.backends.base import (
    DecisionBackend,
    create_hf_zero_shot_scorer,
    create_onnx_nli_scorer,
    semantic_match_logit,
)
from microllm.backends.cascade import DualGearCascadeBackend, fuse_distributions
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.backends.prefill_llm import (
    AnyJevPrefillBackend,
    apply_prior_debiasing,
    compute_answer_mass,
    l0_circular_rotation_readout,
    van_der_corput_order,
)

__all__ = [
    "AnyJevPrefillBackend",
    "DecisionBackend",
    "DualGearCascadeBackend",
    "FastEncoderBackend",
    "apply_prior_debiasing",
    "compute_answer_mass",
    "create_hf_zero_shot_scorer",
    "create_onnx_nli_scorer",
    "format_nli_hypothesis",
    "fuse_distributions",
    "l0_circular_rotation_readout",
    "semantic_match_logit",
    "van_der_corput_order",
]
