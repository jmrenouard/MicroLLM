"""MicroLLM: Local Hybrid System-One Decision Engine (JevCompat 32/32 MUST, AnyJev L0, StartLux Cascade, OpenJev-RLCD)."""

from microllm.backends.base import (
    create_hf_zero_shot_scorer,
    create_onnx_nli_scorer,
    explain_counterfactual,
    explain_state_tokens,
)
from microllm.backends.cascade import (
    DualGearCascadeBackend,
    compute_adaptive_alpha,
    fuse_distributions,
)
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.backends.prefill_llm import (
    AnyJevPrefillBackend,
    OnlinePriorTracker,
    compute_dynamic_tau,
    l0_circular_rotation_readout,
)
from microllm.calibration import (
    PerKTemperatureScaler,
    SplitConformalCalibrator,
    VectorScalingCalibrator,
    brier_score,
    clopper_pearson_threshold,
    expected_calibration_error,
    tune_cascade_pareto,
)
from microllm.client import (
    AsyncMicroJevClient,
    Choice,
    MicroJevClient,
    Multi,
    Noul,
    Score,
    system_one_router,
)
from microllm.confidence import (
    choice_confidence,
    earth_movers_distance_1d,
    multi_confidence,
    score_confidence,
    score_expectation,
    unimodal_ordinal_smooth,
)
from microllm.papers import (
    ACADEMIC_PAPERS_CATALOG,
    LITERATURE_PILLARS,
    format_papers_markdown,
    get_academic_paper,
    get_literature_taxonomy,
    list_academic_papers,
)
from microllm.registry import (
    MODEL_PROFILES,
    dequantize_matrix_int8,
    list_profiles,
    pull_profile,
    quantize_matrix_int8,
    verify_profile,
)
from microllm.rlcd import (
    RLCDConfig,
    RLCDDecisionHeadTrainer,
    RLCDReplayBuffer,
    shifted_brier_reward,
    verify_tabular_rlcd_unbiased_gradient,
)
from microllm.server import create_app

__version__ = "0.2.0"

__all__ = [
    "ACADEMIC_PAPERS_CATALOG",
    "AnyJevPrefillBackend",
    "AsyncMicroJevClient",
    "Choice",
    "DualGearCascadeBackend",
    "FastEncoderBackend",
    "LITERATURE_PILLARS",
    "MODEL_PROFILES",
    "MicroJevClient",
    "Multi",
    "Noul",
    "OnlinePriorTracker",
    "PerKTemperatureScaler",
    "RLCDConfig",
    "RLCDDecisionHeadTrainer",
    "RLCDReplayBuffer",
    "Score",
    "SplitConformalCalibrator",
    "VectorScalingCalibrator",
    "brier_score",
    "choice_confidence",
    "clopper_pearson_threshold",
    "compute_adaptive_alpha",
    "compute_dynamic_tau",
    "create_app",
    "create_hf_zero_shot_scorer",
    "create_onnx_nli_scorer",
    "dequantize_matrix_int8",
    "earth_movers_distance_1d",
    "expected_calibration_error",
    "explain_counterfactual",
    "explain_state_tokens",
    "format_nli_hypothesis",
    "format_papers_markdown",
    "fuse_distributions",
    "get_academic_paper",
    "get_literature_taxonomy",
    "l0_circular_rotation_readout",
    "list_academic_papers",
    "list_profiles",
    "multi_confidence",
    "pull_profile",
    "quantize_matrix_int8",
    "score_confidence",
    "score_expectation",
    "shifted_brier_reward",
    "system_one_router",
    "tune_cascade_pareto",
    "unimodal_ordinal_smooth",
    "verify_profile",
    "verify_tabular_rlcd_unbiased_gradient",
]
