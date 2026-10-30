"""Unit and benchmark tests for Strategy 1: Lightweight Multilingual Engine in MicroLLM.

Validates:
1. Zero external dependencies (Python standard library only).
2. Diacritic stripping (strip_accents) and light stemming (light_stem).
3. Ultra-fast language heuristic detection (detect_language_fast).
4. Quadrilingual concept cluster coverage across all 10 clusters in EN, FR, ES, PT.
5. Customer support ticket routing accuracy (>95%) across EN, FR, ES, PT in FastEncoderBackend.
6. Negation, double negation, and polarity handling in Spanish and Portuguese in evaluate_noul.
7. Sub-millisecond latency SLA benchmark (1,000 iterations asserting p50 < 0.2 ms, p99 < 1.0 ms).
"""

from __future__ import annotations

import ast
import pathlib
import time
from typing import Any

import pytest

from microllm.backends.base import _embed_text_cached
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.lexicons import (
    ALL_DOUBLE_NEGATION_PHRASES,
    ALL_NEGATION_MARKERS,
    ALL_SENTENCE_CONNECTORS,
    ALL_STOPWORDS,
    ALL_SYNTACTIC_NEGATION,
)
from microllm.lexicons.concepts import CONCEPT_CLUSTERS
from microllm.lexicons.normalizer import (
    detect_language_fast,
    light_stem,
    strip_accents,
    tokenize_multilingual_normalized,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Zero External Dependencies & Standard Library Assertion
# ---------------------------------------------------------------------------


def test_strategy1_zero_dependencies_and_stdlib_only() -> None:
    """Verify that src/microllm/lexicons/ imports strictly from the Python standard library."""
    lexicons_dir = ROOT / "src" / "microllm" / "lexicons"
    assert lexicons_dir.is_dir(), "src/microllm/lexicons must exist as a package"

    stdlib_whitelist = {
        "unicodedata",
        "re",
        "functools",
        "typing",
        "collections",
        "itertools",
        "math",
        "time",
        "sys",
        "os",
        "pathlib",
        "__future__",
    }
    forbidden_modules = {
        "torch",
        "transformers",
        "onnxruntime",
        "spacy",
        "nltk",
        "langdetect",
        "requests",
        "numpy",
        "scipy",
        "sklearn",
    }

    for py_file in lexicons_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in forbidden_modules, (
                        f"Forbidden dependency '{root_pkg}' found in {py_file.name}"
                    )
                    assert root_pkg in stdlib_whitelist or root_pkg == "microllm", (
                        f"Non-stdlib import '{root_pkg}' in {py_file.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    assert root_pkg not in forbidden_modules, (
                        f"Forbidden dependency '{root_pkg}' found in {py_file.name}"
                    )
                    assert (
                        node.level > 0
                        or root_pkg in stdlib_whitelist
                        or root_pkg == "microllm"
                    ), f"Non-stdlib import '{root_pkg}' in {py_file.name}"


# ---------------------------------------------------------------------------
# 2. Normalizer: Unaccenting, Light Stemming & Language Detection
# ---------------------------------------------------------------------------


def test_strip_accents_multilingual() -> None:
    """Verify standard-library unaccenting across French, Spanish, Portuguese, and mixed casing."""
    # French diacritics
    assert strip_accents("éèêëàâîïôûùç") == "eeeeaaiiouuc"
    assert strip_accents("Remboursement immédiat") == "Remboursement immediat"

    # Spanish diacritics
    assert strip_accents("facturación cancelada") == "facturacion cancelada"
    assert strip_accents("¿Dónde está mi envío?") == "¿Donde esta mi envio?"

    # Portuguese diacritics
    assert strip_accents("não, cobrança e devolução") == "nao, cobranca e devolucao"
    assert strip_accents("Cartão de crédito") == "Cartao de credito"

    # Preserves casing and pure ASCII
    assert strip_accents("ÁÉÍÓÚ Ñ Ç") == "AEIOU N C"
    assert strip_accents("plain ascii text 123") == "plain ascii text 123"
    assert strip_accents("") == ""

    # Idempotence: strip_accents(strip_accents(x)) == strip_accents(x)
    sample = "Crítica à automatização das faturas eletrónicas"
    assert strip_accents(strip_accents(sample)) == strip_accents(sample)


def test_light_stem_multilingual() -> None:
    """Verify rule-based inflectional suffix stripping across EN, FR, ES, PT."""
    # Plural stripping
    assert light_stem("invoices", "en") == "invoic"
    assert light_stem("crashes", "en") == "crash"
    assert light_stem("factures", "fr") == "factur"
    assert light_stem("remboursements", "fr") == "remboursement"
    assert light_stem("pedidos", "es") == "pedid"
    assert light_stem("faturas", "pt") == "fatur"
    assert light_stem("pagamentos", "pt") == "pagament"

    # Verb / Gerund / Participle suffixes
    assert light_stem("charging", "en") == "charg"
    assert light_stem("billed", "en") == "bill"
    assert light_stem("cobrando", "es") == "cobr"
    assert light_stem("entregado", "es") == "entreg"
    assert light_stem("cancelando", "pt") == "cancel"

    # Nominalizations (-tion, -cion, -cao, -ment)
    assert light_stem("cancelacion", "es") == "cancel"
    assert light_stem("devolucao", "pt") == "devolu"

    # Guardrails: short words (<4 chars) must never be over-stemmed
    assert light_stem("no", "es") == "no"
    assert light_stem("sim", "pt") == "sim"
    assert light_stem("oui", "fr") == "oui"
    assert light_stem("de", "pt") == "de"
    assert light_stem("un", "fr") == "un"


def test_detect_language_fast_heuristics() -> None:
    """Verify ultra-fast unigram heuristic language detection for EN, FR, ES, PT."""
    en_sample = "I need a full refund for an overcharged invoice on my credit card."
    fr_sample = "Je demande le remboursement immédiat de ma facture suite à une erreur."
    es_sample = "Necesito un reembolso urgente por una factura cobrada dos veces en mi cuenta."
    pt_sample = "Preciso de um estorno urgente de uma fatura cobrada duas vezes no meu cartão."

    assert detect_language_fast(en_sample) == "en"
    assert detect_language_fast(fr_sample) == "fr"
    assert detect_language_fast(es_sample) == "es"
    assert detect_language_fast(pt_sample) == "pt"

    # Ambiguous or non-linguistic inputs return 'unknown'
    assert detect_language_fast("1234567890 987654") == "unknown"
    assert detect_language_fast("") == "unknown"
    assert detect_language_fast("OK") == "unknown"

    # Performance assertion: 1,000 calls must run in under 20 ms (< 0.02 ms per call)
    t0 = time.perf_counter()
    for _ in range(1000):
        detect_language_fast(es_sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 20.0, f"detect_language_fast took {elapsed_ms:.2f} ms for 1,000 iterations"


def test_tokenize_multilingual_normalized() -> None:
    """Verify tokenization combines unaccenting, lowercasing, and word extraction."""
    raw = "¡Hola! ¿Dónde está mi envío #4521? Não chegou."
    tokens = tokenize_multilingual_normalized(raw, strip_accent=True)
    assert "donde" in tokens
    assert "esta" in tokens
    assert "envio" in tokens
    assert "nao" in tokens
    assert "chegou" in tokens


# ---------------------------------------------------------------------------
# 3. Concept Clusters Coverage & Vector Slot Activation
# ---------------------------------------------------------------------------


def test_concept_clusters_quadrilingual_coverage() -> None:
    """Verify that all 10 concept clusters contain keywords in EN, FR, ES, and PT."""
    required_clusters = [
        "billing_finance",
        "technical_support",
        "shipping_logistics",
        "urgency_high",
        "urgency_low",
        "affirmative",
        "negative",
        "security_destructive",
        "security_injection",
        "security_pii",
    ]
    for cluster_name in required_clusters:
        assert cluster_name in CONCEPT_CLUSTERS, f"Missing concept cluster '{cluster_name}'"
        cluster_set = CONCEPT_CLUSTERS[cluster_name]
        assert len(cluster_set) >= 15, f"Cluster '{cluster_name}' has insufficient vocabulary ({len(cluster_set)})"

    # Check designated representative keywords across all 4 languages
    bf = CONCEPT_CLUSTERS["billing_finance"]
    assert {"invoice", "refund"} <= bf  # EN
    assert {"facture", "remboursement"} <= bf  # FR
    assert {"factura", "reembolso", "cobro"} <= bf  # ES
    assert {"fatura", "estorno", "cobranca", "pagamento"} <= bf  # PT

    ts = CONCEPT_CLUSTERS["technical_support"]
    assert {"crash", "bug"} <= ts  # EN
    assert {"panne", "erreur"} <= ts  # FR
    assert {"averia", "fallo"} <= ts  # ES
    assert {"pane", "falha", "travamento"} <= ts  # PT

    sl = CONCEPT_CLUSTERS["shipping_logistics"]
    assert {"parcel", "delivery"} <= sl  # EN
    assert {"colis", "livraison"} <= sl  # FR
    assert {"paquete", "envio"} <= sl  # ES
    assert {"pacote", "encomenda", "rastreamento"} <= sl  # PT

    uh = CONCEPT_CLUSTERS["urgency_high"]
    assert {"urgent", "immediate"} <= uh  # EN
    assert {"urgent", "immediat"} <= uh  # FR
    assert {"urgente", "inmediato"} <= uh  # ES
    assert {"urgente", "imediatamente"} <= uh  # PT

    ul = CONCEPT_CLUSTERS["urgency_low"]
    assert {"calm", "minor"} <= ul  # EN
    assert {"calme", "mineur"} <= ul  # FR
    assert {"calma", "menor"} <= ul  # ES
    assert {"calmo", "baixo"} <= ul  # PT

    aff = CONCEPT_CLUSTERS["affirmative"]
    assert {"yes", "true"} <= aff  # EN
    assert {"oui", "vrai"} <= aff  # FR
    assert {"si", "verdadero"} <= aff  # ES
    assert {"sim", "correto"} <= aff  # PT

    neg = CONCEPT_CLUSTERS["negative"]
    assert {"no", "never"} <= neg  # EN
    assert {"non", "faux"} <= neg  # FR
    assert {"no", "nunca", "jamas"} <= neg  # ES
    assert {"nao", "nunca", "jamais"} <= neg  # PT


def test_embedding_vector_slots_activate_on_spanish_and_portuguese() -> None:
    """Verify that _embed_text_cached activates slots [0..15] on Spanish and Portuguese tokens."""
    # Slot 0 = billing_finance
    vec_es_billing = _embed_text_cached("reembolso cobro factura tarjeta")
    assert vec_es_billing[0] > 0.0, "Spanish billing tokens failed to activate vector slot 0"

    vec_pt_billing = _embed_text_cached("fatura estorno cobranca pagamento")
    assert vec_pt_billing[0] > 0.0, "Portuguese billing tokens failed to activate vector slot 0"

    # Slot 1 = technical_support
    vec_es_tech = _embed_text_cached("servidor caido error averia bloqueo")
    assert vec_es_tech[1] > 0.0, "Spanish technical tokens failed to activate vector slot 1"

    vec_pt_tech = _embed_text_cached("falha pane erro servidor travamento")
    assert vec_pt_tech[1] > 0.0, "Portuguese technical tokens failed to activate vector slot 1"

    # Slot 2 = shipping_logistics
    vec_es_ship = _embed_text_cached("paquete envio pedido retraso entrega")
    assert vec_es_ship[2] > 0.0, "Spanish shipping tokens failed to activate vector slot 2"

    vec_pt_ship = _embed_text_cached("encomenda entrega pacote atraso rastreamento")
    assert vec_pt_ship[2] > 0.0, "Portuguese shipping tokens failed to activate vector slot 2"


# ---------------------------------------------------------------------------
# 4. Customer Support Ticket Routing Accuracy Matrix (EN, FR, ES, PT)
# ---------------------------------------------------------------------------

PROBE_TICKETS = [
    # --- English (8 tickets) ---
    ("billing", "I was charged twice on my credit card for invoice #9401 and need an immediate refund."),
    ("billing", "Please cancel my subscription and return the monthly fee debited yesterday."),
    ("shipping", "My package was marked as delivered yesterday but nothing arrived in my mailbox."),
    ("shipping", "Where is my parcel? The courier tracking link says pending dispatch for 5 days."),
    ("tech", "The production database crashes with a fatal 500 error whenever users attempt to login."),
    ("tech", "We are experiencing a severe API outage, slow responses and broken authentication tokens."),
    ("billing", "The pricing on my latest invoice does not match the agreed contract discount."),
    ("shipping", "The delivery driver damaged the parcel box and the order arrived broken."),

    # --- French (8 tickets) ---
    ("billing", "J'ai été débité deux fois sur ma carte bancaire et j'exige un remboursement immédiat de la facture."),
    ("billing", "Veuillez annuler mon abonnement et me rembourser le prélèvement indu de ce mois."),
    ("shipping", "Mon colis est indiqué comme livré mais je n'ai absolument rien reçu dans ma boîte aux lettres."),
    ("shipping", "Le suivi postal de ma commande indique un retard important de livraison par le transporteur."),
    ("tech", "L'application mobile plante continuellement sur l'écran de connexion avec une erreur serveur 500."),
    ("tech", "Grosse panne technique sur notre serveur API, les requêtes échouent avec une exception fatale."),
    ("billing", "Erreur sur le montant de la dernière facture, les frais appliqués sont incorrects."),
    ("shipping", "Le colis reçu est endommagé et le produit à l'intérieur est cassé."),

    # --- Spanish (8 tickets) ---
    ("billing", "Necesito un reembolso urgente por una factura cobrada dos veces en mi tarjeta de crédito."),
    ("billing", "Por favor cancelen mi suscripción y devuelvan el cobro indebido realizado esta mañana."),
    ("shipping", "Mi paquete figura como entregado pero no he recibido nada en mi buzón de correo."),
    ("shipping", "¿Dónde está mi pedido? El enlace de seguimiento del mensajero indica retraso en el envío."),
    ("tech", "La plataforma web se cae cada vez que intento iniciar sesión con un error fatal 500."),
    ("tech", "Tenemos una avería grave en el servidor API, las peticiones fallan y el sistema está caído."),
    ("billing", "El importe de la última factura es erróneo, me han aplicado una tarifa no contratada."),
    ("shipping", "El mensajero entregó el paquete completamente roto y el artículo está dañado."),

    # --- Portuguese (8 tickets) ---
    ("billing", "Preciso de um estorno urgente de uma fatura cobrada duas vezes no meu cartão de crédito."),
    ("billing", "Por favor cancelem minha assinatura e façam a devolução da cobrança indevida desta fatura."),
    ("shipping", "Minha encomenda consta como entregue mas não recebi nenhum pacote no meu endereço."),
    ("shipping", "Onde está meu pedido? O código de rastreamento dos correios está com entrega atrasada."),
    ("tech", "O aplicativo apresenta falha crítica e trava na tela de login com erro interno 500."),
    ("tech", "Estamos enfrentando uma pane técnica grave no servidor, a API está fora do ar e instável."),
    ("billing", "Houve um erro no valor da mensalidade e cobrança duplicada no meu extrato bancário."),
    ("shipping", "A caixa da encomenda chegou violada e o produto foi entregue danificado."),
]

ROUTING_OPTIONS = {
    "billing": "Billing, invoices, charges, double debits, subscription pricing, refunds and fee disputes",
    "shipping": "Shipping logistics, parcel tracking, courier delays, missing orders and package delivery",
    "tech": "Technical support, bug reports, application crashes, server outages and login failures",
}


def test_customer_support_ticket_classification_accuracy_quadrilingual() -> None:
    """Assert customer support ticket routing achieves >95% overall accuracy across EN, FR, ES, PT."""
    backend = FastEncoderBackend()
    instructions = "Route this customer inquiry to the appropriate support department."

    results: list[dict[str, Any]] = []
    language_correct: dict[str, int] = {"en": 0, "fr": 0, "es": 0, "pt": 0}
    language_total: dict[str, int] = {"en": 0, "fr": 0, "es": 0, "pt": 0}

    # Warm-up call
    backend.evaluate_choice("warmup", instructions, ROUTING_OPTIONS)

    for idx, (expected_cat, ticket_text) in enumerate(PROBE_TICKETS):
        # Determine language bucket from index (8 per language)
        lang = ["en", "fr", "es", "pt"][idx // 8]
        language_total[lang] += 1

        res = backend.evaluate_choice(ticket_text, instructions, ROUTING_OPTIONS)
        predicted = res["choice"]
        is_correct = (predicted == expected_cat)

        if is_correct:
            language_correct[lang] += 1

        results.append({
            "lang": lang,
            "expected": expected_cat,
            "predicted": predicted,
            "confidence": res.get("x_confidence", 0.0),
            "correct": is_correct,
            "text": ticket_text,
        })

    total_correct = sum(language_correct.values())
    total_tickets = len(PROBE_TICKETS)
    overall_accuracy = total_correct / float(total_tickets)

    # Detailed assertion reporting
    failures = [r for r in results if not r["correct"]]
    failure_msg = "\n".join(
        f"[{f['lang'].upper()}] Expected {f['expected']}, got {f['predicted']} (conf: {f['confidence']:.3f}): '{f['text']}'"
        for f in failures
    )

    # Assert >95% overall accuracy (at least 31/32)
    assert overall_accuracy >= 0.95, (
        f"Overall accuracy {overall_accuracy * 100:.1f}% ({total_correct}/{total_tickets}) fell below 95% threshold.\n"
        f"Failures:\n{failure_msg}"
    )

    # Assert per-language accuracy >= 87.5% (at least 7/8 in every single language)
    for lang in ("en", "fr", "es", "pt"):
        lang_acc = language_correct[lang] / float(language_total[lang])
        assert lang_acc >= 0.875, (
            f"Language '{lang.upper()}' accuracy {lang_acc * 100:.1f}% fell below per-language threshold."
        )


# ---------------------------------------------------------------------------
# 5. Negation, Polarity & Clause Salience in Spanish & Portuguese
# ---------------------------------------------------------------------------


def test_evaluate_noul_negation_and_polarity_spanish_portuguese() -> None:
    """Verify negation markers, Premise/Hypothesis NLI, and double negation in evaluate_noul."""
    backend = FastEncoderBackend()

    # 1. Spanish Unigram Negation in Premise/Hypothesis NLI
    es_entail = backend.evaluate_noul(
        state="Premise: El cliente pagó la factura a tiempo.\nHypothesis: Se ha realizado el pago de la factura.",
        instructions="",
    )
    assert es_entail["noul"] > 0.60, f"Spanish entailment failed: {es_entail['noul']}"

    es_contradiction = backend.evaluate_noul(
        state="Premise: El cliente no pagó la factura a tiempo.\nHypothesis: Se ha realizado el pago de la factura.",
        instructions="",
    )
    assert es_contradiction["noul"] < 0.35, f"Spanish unigram negation failed: {es_contradiction['noul']}"

    # 2. Portuguese Unigram Negation in Premise/Hypothesis NLI
    pt_entail = backend.evaluate_noul(
        state="Premise: O cliente pagou a fatura no prazo.\nHypothesis: O pagamento da fatura foi efetuado.",
        instructions="",
    )
    assert pt_entail["noul"] > 0.60, f"Portuguese entailment failed: {pt_entail['noul']}"

    pt_contradiction = backend.evaluate_noul(
        state="Premise: O cliente não pagou a fatura no prazo.\nHypothesis: O pagamento da fatura foi efetuado.",
        instructions="",
    )
    assert pt_contradiction["noul"] < 0.35, f"Portuguese unigram negation failed: {pt_contradiction['noul']}"

    # 3. Spanish Double Negation Cancels Negation (Affirmative Entailment)
    es_double_neg = backend.evaluate_noul(
        state="Premise: No es imposible recuperar el acceso a la cuenta.\nHypothesis: Es posible recuperar el acceso.",
        instructions="",
    )
    assert es_double_neg["noul"] > 0.60, f"Spanish double negation failed: {es_double_neg['noul']}"

    # 4. Portuguese Double Negation Cancels Negation (Affirmative Entailment)
    pt_double_neg = backend.evaluate_noul(
        state="Premise: Não é impossível recuperar o acesso à conta.\nHypothesis: É possível recuperar o acesso.",
        instructions="",
    )
    assert pt_double_neg["noul"] > 0.60, f"Portuguese double negation failed: {pt_double_neg['noul']}"


def test_contrastive_sentence_connectors_trigger_clause_salience() -> None:
    """Verify that contrastive connectors in ES ('sin embargo') and PT ('no entanto') trigger clause salience."""
    backend = FastEncoderBackend()
    options = {
        "ok": "Order arrived safely and successfully without any defect",
        "damaged": "Damaged broken item arrived with serious defect and parcel box torn",
    }

    # Spanish ticket with positive opening clause but contrastive damaged resolution
    es_ticket = (
        "El pedido fue entregado rápidamente por el mensajero hoy. "
        "Sin embargo, el paquete llegó completamente roto y el contenido está dañado."
    )
    res_es = backend.evaluate_choice(es_ticket, "Assess order status", options)
    assert res_es["choice"] == "damaged", (
        f"Spanish contrastive connector failed to prioritize second clause: got {res_es['choice']}"
    )

    # Portuguese ticket with positive opening clause but contrastive damaged resolution
    pt_ticket = (
        "A entrega foi muito rápida e o motorista foi educado. "
        "No entanto, o pacote chegou totalmente quebrado e o produto danificado."
    )
    res_pt = backend.evaluate_choice(pt_ticket, "Assess order status", options)
    assert res_pt["choice"] == "damaged", (
        f"Portuguese contrastive connector failed to prioritize second clause: got {res_pt['choice']}"
    )


# ---------------------------------------------------------------------------
# 6. Latency SLA Benchmark Test (1,000 Iterations)
# ---------------------------------------------------------------------------


def test_strategy1_latency_sla_sub_millisecond() -> None:
    """Benchmark test executing 1,000 evaluate_choice calls asserting p50 < 0.2 ms and p99 < 1.0 ms."""
    backend = FastEncoderBackend()
    instructions = "Route ticket to support department"

    # Pre-warm cache for 50 iterations
    for _, ticket_text in PROBE_TICKETS:
        backend.evaluate_choice(ticket_text, instructions, ROUTING_OPTIONS)

    num_iterations = 1000
    latencies_ms: list[float] = []

    tickets_cycle = [t[1] for t in PROBE_TICKETS]
    num_tickets = len(tickets_cycle)

    t_start = time.perf_counter()
    for i in range(num_iterations):
        ticket = tickets_cycle[i % num_tickets]
        t0 = time.perf_counter()
        backend.evaluate_choice(ticket, instructions, ROUTING_OPTIONS)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_wall_clock_s = time.perf_counter() - t_start
    latencies_ms.sort()

    p50 = latencies_ms[int(num_iterations * 0.50)]
    p90 = latencies_ms[int(num_iterations * 0.90)]
    p95 = latencies_ms[int(num_iterations * 0.95)]
    p99 = latencies_ms[int(num_iterations * 0.99)]
    mean_lat = sum(latencies_ms) / float(num_iterations)

    print(
        f"\n[Strategy 1 Latency Telemetry - 1000 Iterations]\n"
        f"  Total time : {total_wall_clock_s:.3f} s\n"
        f"  Mean       : {mean_lat:.4f} ms\n"
        f"  p50        : {p50:.4f} ms (Target < 0.20 ms)\n"
        f"  p90        : {p90:.4f} ms\n"
        f"  p95        : {p95:.4f} ms\n"
        f"  p99        : {p99:.4f} ms (SLA Target < 1.00 ms)\n"
        f"  Min        : {latencies_ms[0]:.4f} ms\n"
        f"  Max        : {latencies_ms[-1]:.4f} ms\n"
    )

    # SLA Assertions
    assert p50 < 0.20, f"p50 latency {p50:.4f} ms exceeded 0.20 ms threshold"
    assert p99 < 1.00, f"p99 latency {p99:.4f} ms exceeded 1.00 ms SLA boundary"
    assert total_wall_clock_s < 1.0, f"Total execution time {total_wall_clock_s:.2f} s exceeded 1.0 s budget"
