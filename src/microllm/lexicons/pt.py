"""Portuguese lexicon specifications: stopwords, negation markers, phrases, and connectors."""

from __future__ import annotations

# Exact frozen set of Portuguese stopwords (60 items: between 30 and 60)
STOPWORDS: frozenset[str] = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do",
    "da", "dos", "das", "em", "no", "na", "nos", "nas", "ao", "aos",
    "por", "pelo", "pela", "pelos", "pelas", "para", "com", "e", "ou",
    "que", "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "seu", "sua", "seus", "suas", "meu", "minha", "meus", "minhas", "nosso", "nossa",
    "era", "foi", "foram", "ser", "estar", "como", "mas", "se", "ele", "ela", "eles", "elas",
})

# Localized Portuguese negation markers (30 items)
NEGATION_MARKERS: frozenset[str] = frozenset({
    "nao", "não", "nunca", "jamais", "sem", "nenhum", "nenhuma", "nenhuns", "nenhumas",
    "ninguem", "ninguém", "nada", "tampouco", "nem", "falso", "falsa",
    "recusado", "recusada", "rejeitado", "rejeitada", "impossivel", "impossível",
    "negado", "negada", "proibido", "proibida", "ausente", "invalido", "inválido",
    "falho", "falha",
})

# Core syntactic negation words inverting premise/hypothesis truth polarity (23 items)
SYNTACTIC_NEGATION: frozenset[str] = frozenset({
    "nao", "não", "nunca", "jamais", "sem", "nenhum", "nenhuma",
    "ninguem", "ninguém", "nada", "tampouco", "nem", "falso", "falsa",
    "recusado", "recusada", "rejeitado", "rejeitada", "impossivel", "impossível",
    "negado", "negada", "proibido", "proibida", "invalido", "inválido",
})

# Double negation idioms canceling polarity (12 items)
DOUBLE_NEGATION_PHRASES: tuple[str, ...] = (
    "nao impossivel", "não impossível", "não é impossível", "nao e impossivel",
    "nao improvavel", "não improvável", "nunca negado", "nunca negada",
    "nunca recusado", "nao proibido", "não proibido", "sem nenhuma duvida",
)

# Contrastive sentence connectors triggering clause salience (10 items)
SENTENCE_CONNECTORS: tuple[str, ...] = (
    "no entanto", "no entanto,", "contudo", "contudo,", "todavia", "todavia,",
    "entretanto", "entretanto,", "enquanto isso", "por outro lado",
)

# Natural language inference hypothesis templates
HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "default": "Trata-se de {option}.",
    "with_desc": "Trata-se de {option}: {description}.",
}
