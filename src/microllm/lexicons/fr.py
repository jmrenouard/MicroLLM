"""French lexicon specifications: stopwords, negation markers, phrases, and connectors."""

from __future__ import annotations

# Exact frozen set of French stopwords (54 items: between 30 and 60)
STOPWORDS: frozenset[str] = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "et", "ou", "a", "au", "aux", "en", "dans", "par", "pour", "sur",
    "avec", "qui", "que", "dont", "est", "sont", "ete", "etre",
    "avoir", "ont", "il", "elle", "on", "nous", "vous", "ils", "elles",
})

# Localized French negation markers (30 items)
NEGATION_MARKERS: frozenset[str] = frozenset({
    "ne", "non", "pas", "aucun", "aucune", "aucuns", "aucunes", "jamais", "rien",
    "personne", "nul", "nulle", "sans", "ni", "point", "guere", "refuse", "refusee",
    "rejete", "rejetee", "impossible", "faux", "fausse", "interdit", "interdite",
    "manquant", "manquante", "invalide", "absent", "absente",
})

# Core syntactic negation words inverting premise/hypothesis truth polarity (21 items)
SYNTACTIC_NEGATION: frozenset[str] = frozenset({
    "non", "pas", "aucun", "aucune", "jamais", "rien", "personne", "nul", "nulle",
    "sans", "ni", "refuse", "refusee", "rejete", "rejetee", "impossible", "faux",
    "fausse", "interdit", "interdite", "invalide",
})

# Double negation idioms canceling polarity (11 items)
DOUBLE_NEGATION_PHRASES: tuple[str, ...] = (
    "pas impossible", "n'est pas impossible", "pas improbable", "nullement impossible",
    "jamais refuse", "jamais refusee", "jamais rejete", "jamais rejetee",
    "non interdit", "pas faux", "sans aucun doute",
)

# Contrastive sentence connectors triggering clause salience (12 items)
SENTENCE_CONNECTORS: tuple[str, ...] = (
    "cependant", "cependant,", "toutefois", "toutefois,", "neanmoins", "néanmoins",
    "pourtant", "en revanche", "d un autre cote", "d'un autre côté",
    "mais", "mais,",
)

# Natural language inference hypothesis templates
HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "default": "Il s'agit de {option}.",
    "with_desc": "Il s'agit de {option}: {description}.",
}
