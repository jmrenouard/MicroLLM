"""Spanish lexicon specifications: stopwords, negation markers, phrases, and connectors."""

from __future__ import annotations

# Exact frozen set of Spanish stopwords (56 items: between 30 and 60)
STOPWORDS: frozenset[str] = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "al", "a", "en", "por", "para", "con", "y", "e", "o", "u",
    "que", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "nuestros", "nuestras",
    "es", "son", "era", "eran", "fue", "fueron", "ser", "estar", "ha", "han",
    "como", "pero", "si", "yo", "ella", "nosotros", "ellos",
})

# Localized Spanish negation markers (30 items)
NEGATION_MARKERS: frozenset[str] = frozenset({
    "no", "nunca", "jamas", "jamás", "sin", "ninguno", "ninguna", "ningun", "ningún",
    "ningunos", "ningunas", "nadie", "nada", "tampoco", "ni", "falso", "falsa",
    "rechazado", "rechazada", "imposible", "denegado", "denegada", "prohibido",
    "prohibida", "ausente",
    "invalido", "inválido", "invalida", "inválida",
    "fallido", "fallida",
})

# Core syntactic negation words inverting premise/hypothesis truth polarity (26 items)
SYNTACTIC_NEGATION: frozenset[str] = frozenset({
    "no", "nunca", "jamas", "jamás", "sin", "ninguno", "ninguna", "ningun", "ningún",
    "nadie", "nada", "tampoco", "ni", "falso", "falsa", "rechazado", "rechazada",
    "imposible", "denegado", "denegada", "prohibido", "prohibida",
    "invalido", "inválido", "invalida", "inválida",
})

# Double negation idioms canceling polarity (14 items)
DOUBLE_NEGATION_PHRASES: tuple[str, ...] = (
    "no imposible", "no es imposible", "no improbable", "no es improbable",
    "nunca denegado", "nunca denegada", "nunca rechazado", "nunca rechazada",
    "no prohibido", "no esta prohibido", "no está prohibido", "no es prohibido",
    "no falso", "sin ninguna duda",
)

# Contrastive sentence connectors triggering clause salience (8 items)
SENTENCE_CONNECTORS: tuple[str, ...] = (
    "sin embargo", "sin embargo,", "no obstante", "no obstante,",
    "mientras tanto", "por el contrario", "en cambio", "a pesar de ello",
)

# Natural language inference hypothesis templates
HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "default": "Se trata de {option}.",
    "with_desc": "Se trata de {option}: {description}.",
}
