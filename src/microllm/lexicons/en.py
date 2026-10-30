"""English lexicon specifications: stopwords, negation markers, phrases, and connectors."""

from __future__ import annotations

# Exact frozen set of English stopwords (58 items: between 30 and 60)
STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "having", "he", "her", "his", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "should", "so", "than",
    "that", "the", "their", "them", "there", "they", "this", "to", "was", "we",
    "were", "which", "who", "will", "with", "would", "you", "your",
})

# Localized English negation markers (31 items)
NEGATION_MARKERS: frozenset[str] = frozenset({
    "no", "not", "never", "none", "neither", "nor", "nobody", "nothing", "nowhere",
    "cannot", "cant", "without", "hardly", "barely", "scarcely", "rarely", "seldom",
    "absent", "missing", "denied", "refused", "rejected", "failed", "prohibited",
    "forbidden", "impossible", "unlikely", "unverified", "false", "opposite", "violation",
})

# Core syntactic negation words inverting premise/hypothesis truth polarity (24 items)
SYNTACTIC_NEGATION: frozenset[str] = frozenset({
    "no", "not", "never", "none", "neither", "nor", "nobody", "nothing", "nowhere",
    "cannot", "without", "denied", "refused", "rejected", "failed", "prohibited",
    "forbidden", "impossible", "false", "opposite", "unlikely", "hardly", "barely", "scarcely",
})

# Double negation idioms canceling polarity (15 items)
DOUBLE_NEGATION_PHRASES: tuple[str, ...] = (
    "not impossible", "not unlikely", "not uncommon", "not unusual",
    "never denied", "never refused", "never failed", "never fails",
    "not prohibited", "not forbidden", "not rejected", "not refused",
    "not false", "not invalid", "without any doubt",
)

# Contrastive sentence connectors triggering clause salience (9 items)
SENTENCE_CONNECTORS: tuple[str, ...] = (
    "however,", "however", "meanwhile", "nevertheless", "nonetheless",
    "on the other hand", "yet", "although", "even though",
)

# Natural language inference hypothesis templates
HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "default": "It is {option}.",
    "with_desc": "It is {option}: {description}.",
}
