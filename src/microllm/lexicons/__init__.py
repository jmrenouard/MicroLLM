"""Unified multilingual lexicons package for MicroLLM Strategy 1.

Exports aggregate frozen sets across English, French, Spanish, and Portuguese,
alongside normalizer utilities and per-language modules.
"""

from __future__ import annotations

from microllm.lexicons import en, es, fr, pt
from microllm.lexicons.normalizer import (
    detect_language_fast,
    light_stem,
    strip_accents,
    tokenize_multilingual_normalized,
)

# Aggregated frozen sets across all 4 supported languages
ALL_STOPWORDS: frozenset[str] = frozenset().union(
    en.STOPWORDS,
    fr.STOPWORDS,
    es.STOPWORDS,
    pt.STOPWORDS,
)

ALL_NEGATION_MARKERS: frozenset[str] = frozenset().union(
    en.NEGATION_MARKERS,
    fr.NEGATION_MARKERS,
    es.NEGATION_MARKERS,
    pt.NEGATION_MARKERS,
)

ALL_SYNTACTIC_NEGATION: frozenset[str] = frozenset().union(
    en.SYNTACTIC_NEGATION,
    fr.SYNTACTIC_NEGATION,
    es.SYNTACTIC_NEGATION,
    pt.SYNTACTIC_NEGATION,
)

ALL_DOUBLE_NEGATION_PHRASES: tuple[str, ...] = (
    en.DOUBLE_NEGATION_PHRASES
    + fr.DOUBLE_NEGATION_PHRASES
    + es.DOUBLE_NEGATION_PHRASES
    + pt.DOUBLE_NEGATION_PHRASES
)

ALL_SENTENCE_CONNECTORS: tuple[str, ...] = (
    en.SENTENCE_CONNECTORS
    + fr.SENTENCE_CONNECTORS
    + es.SENTENCE_CONNECTORS
    + pt.SENTENCE_CONNECTORS
)

LANGUAGE_MODULES = {
    "en": en,
    "fr": fr,
    "es": es,
    "pt": pt,
}

NLI_TEMPLATES = {
    "en": en.HYPOTHESIS_TEMPLATES,
    "fr": fr.HYPOTHESIS_TEMPLATES,
    "es": es.HYPOTHESIS_TEMPLATES,
    "pt": pt.HYPOTHESIS_TEMPLATES,
}

__all__ = [
    "ALL_DOUBLE_NEGATION_PHRASES",
    "ALL_NEGATION_MARKERS",
    "ALL_SENTENCE_CONNECTORS",
    "ALL_STOPWORDS",
    "ALL_SYNTACTIC_NEGATION",
    "LANGUAGE_MODULES",
    "NLI_TEMPLATES",
    "detect_language_fast",
    "en",
    "es",
    "fr",
    "light_stem",
    "pt",
    "strip_accents",
    "tokenize_multilingual_normalized",
]
