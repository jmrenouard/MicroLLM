"""Empirical Adversarial Challenge Harness for Milestone 2 - Challenger 2.

Adversarially tests:
1. Extreme inputs and edge cases for `format_nli_hypothesis`:
   - Empty strings, pure whitespace, special characters, formatting injections ({name}, {desc}).
   - Pre-formatted prefix deduplication across all 4 languages (EN, FR, ES, PT) and styles (default, about, inquiry).
   - Emojis, accents, code snippets, and long payloads.
   - Trailing punctuation and case insensitivity.
2. Cross-lingual code-switching:
   - Spanglish, Portunhol, Franglais, and technical loanwords.
   - Dynamic language inference via `detect_language_fast` in hypothesis formatting and routing.
3. Multi-template log-space consensus (`HYP-011`) in French, Spanish, and Portuguese:
   - Verification of localized consensus templates at narrow margins (<= 0.12).
   - Detection of IEEE 754 floating-point boundary sensitivity on exact margin 0.12.
   - Consensus behavior with missing/empty option descriptions.
   - Consensus resolution of ambiguous candidate margins in native languages.
4. Localized negative anchor scoring (`_NEGATIVE_HYPOTHESES`) in `FastEncoderBackend.evaluate_multi`:
   - Localized negative anchor hypothesis dispatch for EN, FR, ES, PT.
   - Neural NLI negative anchor suppression of completely irrelevant inputs.
   - Neural NLI negative anchor suppression under explicit rejection/negation inputs.
   - Joint multi-label activation on dual-intent inputs.
   - Behavior analysis under lightweight CPU semantic match logit baseline.
5. End-to-end decision routing under stress:
   - Large option set tournament (>25 options) with localized templates.
   - Ordinal scoring with localized levels.
   - Rubric conflict detection (HYP-006) with localized templates.
"""

from __future__ import annotations

import numpy as np
import pytest

from microllm.backends.fast_encoder import (
    FastEncoderBackend,
    _CONSENSUS_TEMPLATES,
    _NEGATIVE_HYPOTHESES,
    _NLI_TEMPLATES_BY_STYLE,
    format_nli_hypothesis,
)
from microllm.lexicons.normalizer import detect_language_fast


# ===========================================================================
# 1. Extreme Inputs & Formatting Injections for format_nli_hypothesis
# ===========================================================================


def test_format_nli_hypothesis_empty_and_whitespace_inputs() -> None:
    """Challenge format_nli_hypothesis with empty strings, whitespace, and empty descriptions."""
    # Empty option name and no description
    assert format_nli_hypothesis("", None, language="en") == "It is ."
    assert format_nli_hypothesis("   ", None, language="en") == "It is ."
    assert format_nli_hypothesis("\t\n  \r", None, language="en") == "It is ."

    # Empty option name in other languages
    assert format_nli_hypothesis("", None, language="fr") == "Il s'agit de ."
    assert format_nli_hypothesis("", None, language="es") == "Se trata de ."
    assert format_nli_hypothesis("", None, language="pt") == "Trata-se de ."

    # Valid name with empty or whitespace description (should not have colon)
    assert format_nli_hypothesis("billing", "", language="en") == "It is billing."
    assert format_nli_hypothesis("facturation", "   ", language="fr") == "Il s'agit de facturation."
    assert format_nli_hypothesis("facturación", "\n\t", language="es") == "Se trata de facturación."
    assert format_nli_hypothesis("faturamento", None, language="pt") == "Trata-se de faturamento."


def test_format_nli_hypothesis_formatting_injection_safety() -> None:
    """Verify that Python format strings inside option name or description do NOT cause KeyErrors."""
    injections = [
        ("{name}", "{desc}"),
        ("{0}", "{1}"),
        ("{{nested}}", "{{escaped}}"),
        ("{unknown_var}", "{another_unknown}"),
        ("format_%s_%d", "desc_%s"),
    ]

    for name_inj, desc_inj in injections:
        hyp = format_nli_hypothesis(name_inj, desc_inj, language="en")
        assert name_inj in hyp, f"Expected {name_inj} in formatted hypothesis {hyp}"
        assert desc_inj in hyp, f"Expected {desc_inj} in formatted hypothesis {hyp}"

        # Same test across French, Spanish, Portuguese
        hyp_fr = format_nli_hypothesis(name_inj, desc_inj, language="fr")
        assert hyp_fr.startswith("Il s'agit de ")
        hyp_es = format_nli_hypothesis(name_inj, desc_inj, language="es")
        assert hyp_es.startswith("Se trata de ")
        hyp_pt = format_nli_hypothesis(name_inj, desc_inj, language="pt")
        assert hyp_pt.startswith("Trata-se de ")


def test_format_nli_hypothesis_special_characters_and_emojis() -> None:
    """Verify unicode emojis, accented characters, and special symbols are preserved verbatim."""
    special_cases = [
        ("🎉 Promotion", "Réduction spéciale 50% sur l'abonnement", "fr"),
        ("💳 Facturación", "¿Dónde está mi reembolso de $150.00?", "es"),
        ("📦 Encomenda", "Código de rastreamento: #BR-98472-X!", "pt"),
        ("🔥 Bug & Crash", "HTTP 500: Server error @ api.domain.com/v1", "en"),
        ("user@domain.org", "Support inquiries from VIP tier", "en"),
    ]

    for name, desc, lang in special_cases:
        hyp = format_nli_hypothesis(name, desc, language=lang)
        assert name in hyp
        assert desc.rstrip(".") in hyp
        assert hyp.endswith(".")


def test_format_nli_hypothesis_preformatted_prefix_deduplication_all_styles_all_langs() -> None:
    """Verify prefix deduplication across all 4 languages and all 3 template styles."""
    styles = ["default", "about", "inquiry"]
    langs = ["en", "fr", "es", "pt"]

    for style in styles:
        for lang in langs:
            prefix = _NLI_TEMPLATES_BY_STYLE[style][lang][0]
            # Already formatted input without description
            raw_input = f"{prefix} billing"
            hyp = format_nli_hypothesis(raw_input, None, language=lang, template_type=style)
            # Must NOT duplicate prefix
            assert not hyp.lower().startswith(f"{prefix.lower()} {prefix.lower()}"), (
                f"Duplicate prefix found: {hyp} for style={style}, lang={lang}"
            )
            assert hyp.endswith(".")

            # Already formatted input with description
            hyp_with_desc = format_nli_hypothesis(raw_input, "refund inquiries", language=lang, template_type=style)
            assert not hyp_with_desc.lower().startswith(f"{prefix.lower()} {prefix.lower()}")
            assert "refund inquiries" in hyp_with_desc


def test_format_nli_hypothesis_code_snippets_and_long_payloads() -> None:
    """Verify that source code snippets and long payloads format cleanly without breaking."""
    code_samples = [
        ("def calc_loss(y, p):", "return -np.sum(y * np.log(p))"),
        ("SELECT * FROM users;", "WHERE active = 1 AND balance > 0"),
        ("curl -X POST https://api.com/v1", "headers: Authorization: Bearer token123"),
        ("<h1>Title</h1>", "<script>alert('xss')</script>"),
    ]

    for code_name, code_desc in code_samples:
        hyp = format_nli_hypothesis(code_name, code_desc, language="en")
        assert code_name in hyp
        assert code_desc in hyp

    # 1000-character payload
    long_name = "label_" + "x" * 1000
    long_desc = "description_" + "y" * 1500
    long_hyp = format_nli_hypothesis(long_name, long_desc, language="en")
    assert long_name in long_hyp
    assert long_desc in long_hyp


def test_format_nli_hypothesis_case_insensitivity_and_unknown_fallbacks() -> None:
    """Verify case normalization for language and template_type, plus fallback for unknown values."""
    # Case normalization
    assert format_nli_hypothesis("billing", "refunds", language="  FR  ").startswith("Il s'agit de")
    assert format_nli_hypothesis("billing", "refunds", language="Es\t").startswith("Se trata de")
    assert format_nli_hypothesis("billing", "refunds", language="pT").startswith("Trata-se de")

    # Unknown language falls back to English
    assert format_nli_hypothesis("billing", "refunds", language="de").startswith("It is")
    assert format_nli_hypothesis("billing", "refunds", language="zh").startswith("It is")
    assert format_nli_hypothesis("billing", "refunds", language="unknown_dialect").startswith("It is")
    assert format_nli_hypothesis("billing", "refunds", language=None).startswith("It is")

    # Unknown template_type falls back to default style
    assert format_nli_hypothesis("billing", "refunds", language="fr", template_type="nonexistent").startswith("Il s'agit de")


def test_format_nli_hypothesis_trailing_period_observation() -> None:
    """Empirical observation: option_name with trailing period retains period before template dot."""
    hyp_period = format_nli_hypothesis("billing.", None, language="en")
    # Clean name retains period, resulting in 'It is billing..'
    assert hyp_period == "It is billing.."

    # With description, period from clean_name stays before colon
    hyp_with_desc = format_nli_hypothesis("billing.", "refunds.", language="en")
    assert hyp_with_desc == "It is billing.: refunds."


# ===========================================================================
# 2. Cross-Lingual Code-Switching & Dynamic Language Inference
# ===========================================================================


def test_code_switching_spanglish_hypothesis_formatting_and_routing() -> None:
    """Stress-test Spanglish inquiries in format_nli_hypothesis and routing."""
    spanglish_tickets = [
        ("Hola I need a refund for my subscription.", "billing"),
        ("Hola amigo, could you please check my package tracking?", "shipping"),
        ("Quiero saber why was I charged twice on my credit card.", "billing"),
        ("Tengo un bug on my dashboard and cannot login to server.", "technical"),
        ("I need a refund pero la app dice que no es posible.", "billing"),
    ]

    categories = {
        "billing": "Billing, refunds, double charges, subscriptions",
        "shipping": "Shipping logistics, parcels, courier delivery",
        "technical": "Technical support, server errors, login failures",
    }

    for ticket, expected_cat in spanglish_tickets:
        hyp = format_nli_hypothesis("billing", "refunds", premise=ticket)
        assert hyp.endswith(".")

        backend = FastEncoderBackend()
        res = backend.evaluate_choice(ticket, "Route ticket", categories)
        assert res["choice"] == expected_cat, (
            f"Spanglish routing failed for '{ticket}': expected {expected_cat}, got {res['choice']}"
        )


def test_code_switching_portunhol_hypothesis_formatting_and_routing() -> None:
    """Stress-test Portunhol inquiries in format_nli_hypothesis and routing."""
    portunhol_tickets = [
        ("Quiero meu dinheiro de volta por favor, cobrança indevida.", "billing"),
        ("Por favor mi paquete não foi entregue e o rastreio sumiu.", "shipping"),
        ("El servidor está com falha de conexão e erro 500 no login.", "technical"),
        ("Gostaria de saber quando llega mi pedido atrasado.", "shipping"),
    ]

    categories = {
        "billing": "Faturamento, cobrança, reembolsos e pagamentos",
        "shipping": "Envio, encomendas, rastreamento e entrega",
        "technical": "Suporte técnico, falhas no servidor, erros de login",
    }

    for ticket, expected_cat in portunhol_tickets:
        hyp = format_nli_hypothesis("faturamento", "reembolsos", premise=ticket)
        assert hyp.endswith(".")

        backend = FastEncoderBackend()
        res = backend.evaluate_choice(ticket, "Classificar ticket", categories)
        assert res["choice"] == expected_cat, (
            f"Portunhol routing failed for '{ticket}': expected {expected_cat}, got {res['choice']}"
        )


def test_code_switching_franglais_hypothesis_formatting_and_routing() -> None:
    """Stress-test Franglais inquiries in format_nli_hypothesis and routing."""
    franglais_tickets = [
        ("Bonjour where is my package tracked, livraison en retard.", "shipping"),
        ("J'ai un bug on my dashboard and cannot login, erreur 500.", "technical"),
        ("Please do a refund sur ma carte bancaire, double prélèvement.", "billing"),
        ("Mon colis is stuck in customs douane depuis une semaine.", "shipping"),
    ]

    categories = {
        "billing": "Facturation, prélèvements, remboursements et cartes bancaires",
        "shipping": "Livraison, colis, suivi de transport et douane",
        "technical": "Support technique, pannes de serveur, bugs et erreurs de connexion",
    }

    for ticket, expected_cat in franglais_tickets:
        hyp = format_nli_hypothesis("facturation", "remboursements", premise=ticket)
        assert hyp.endswith(".")

        backend = FastEncoderBackend()
        res = backend.evaluate_choice(ticket, "Aiguiller ticket", categories)
        assert res["choice"] == expected_cat, (
            f"Franglais routing failed for '{ticket}': expected {expected_cat}, got {res['choice']}"
        )


def test_technical_loanwords_preserves_target_language_detection() -> None:
    """Verify that English technical loanwords embedded in FR, ES, PT do not falsely flip language detection to English."""
    loanword_cases = [
        ("FR", "Erreur 500 sur le webhook Stripe lors du checkout de l'application.", "fr"),
        ("ES", "Falla de timeout en el API gateway durante el login del usuario.", "es"),
        ("PT", "Problema no deploy do container Docker na cloud AWS.", "pt"),
    ]

    for label, text, expected_lang in loanword_cases:
        detected = detect_language_fast(text)
        assert detected == expected_lang, (
            f"Loanword case {label} falsely classified as {detected}, expected {expected_lang}"
        )
        hyp = format_nli_hypothesis("technique", "erreurs", language=detected)
        prefix = _NLI_TEMPLATES_BY_STYLE["default"][expected_lang][0]
        assert hyp.startswith(prefix)


# ===========================================================================
# 3. Multi-Template Consensus Stress in French, Spanish, Portuguese (HYP-011)
# ===========================================================================


def test_consensus_triggers_at_narrow_margins_quadrilingual() -> None:
    """Verify consensus triggers when margin is narrow (e.g. 0.08 <= 0.12) and is skipped when margin > 0.12."""
    recorded_calls: list[str] = []

    def mock_margin_scorer(target_margin: float):
        def _scorer(state: str, inst: str, hyp: str) -> float:
            recorded_calls.append(hyp)
            if "billing" in hyp or "factur" in hyp or "fatur" in hyp:
                return 1.00 + target_margin
            elif "shipping" in hyp or "livr" in hyp or "env" in hyp:
                return 1.00
            return -1.0
        return _scorer

    # Test 1: margin = 0.08 (<= 0.12) -> MUST trigger consensus in French
    recorded_calls.clear()
    b_narrow = FastEncoderBackend(logit_scorer=mock_margin_scorer(0.08))
    b_narrow.evaluate_choice(
        "Mon colis est en retard et problème de livraison.",
        "Classer",
        {"billing": "Facturation", "shipping": "Livraison"},
    )
    # If consensus triggered: 2 initial calls + 4 consensus calls = 6 calls
    assert len(recorded_calls) == 6, f"Consensus should trigger at margin=0.08, got {len(recorded_calls)} calls"
    assert any("La classification correcte est" in c for c in recorded_calls)
    assert any("Cet état correspond à" in c for c in recorded_calls)

    # Test 2: margin = 0.15 (> 0.12) -> MUST NOT trigger consensus
    recorded_calls.clear()
    b_wide = FastEncoderBackend(logit_scorer=mock_margin_scorer(0.15))
    b_wide.evaluate_choice(
        "Mon colis est en retard et problème de livraison.",
        "Classer",
        {"billing": "Facturation", "shipping": "Livraison"},
    )
    assert len(recorded_calls) == 2, f"Consensus should NOT trigger at margin=0.15, got {len(recorded_calls)} calls"


def test_consensus_ieee754_exact_boundary_empirical_behavior() -> None:
    """Empirically document that 1.12 - 1.00 = 0.1200000000000001 in IEEE 754, testing exact <= 0.12 boundary."""
    recorded_calls: list[str] = []

    def mock_boundary_scorer(state: str, inst: str, hyp: str) -> float:
        recorded_calls.append(hyp)
        if "billing" in hyp:
            return 1.12  # in float64, 1.12 - 1.00 = 0.1200000000000001
        return 1.00

    backend = FastEncoderBackend(logit_scorer=mock_boundary_scorer)
    backend.evaluate_choice(
        "Mon colis est en retard et problème de livraison.",
        "Classer",
        {"billing": "Facturation", "shipping": "Livraison"},
    )

    # In Python IEEE 754 float: 1.12 - 1.00 = 0.1200000000000001 > 0.12.
    # Therefore, consensus is skipped on unrounded float margins exceeding 0.12 by 1e-16.
    margin_val = 1.12 - 1.00
    assert margin_val > 0.12
    assert len(recorded_calls) == 2, "Consensus skipped due to float delta 0.1200000000000001 > 0.12"


def test_consensus_quadrilingual_templates_accuracy() -> None:
    """Verify that consensus uses the exact localized templates for EN, FR, ES, PT."""
    cases = [
        ("en", "I have an issue with my parcel delivery.", "The correct classification is", "This state corresponds to"),
        ("fr", "J'ai un problème avec ma livraison de colis.", "La classification correcte est", "Cet état correspond à"),
        ("es", "Tengo un problema con la entrega de mi paquete.", "La clasificación correcta es", "Este estado corresponde a"),
        ("pt", "Tenho um problema com a entrega da minha encomenda.", "A classificação correta é", "Este estado corresponde a"),
    ]

    for lang, premise, exp_tpl1, exp_tpl2 in cases:
        calls: list[str] = []

        def mock_scorer(state: str, inst: str, hyp: str) -> float:
            calls.append(hyp)
            # Create narrow margin (0.05 <= 0.12)
            if "shipping" in hyp or "colis" in hyp or "paquete" in hyp or "encomenda" in hyp:
                return 1.05
            return 1.00

        backend = FastEncoderBackend(logit_scorer=mock_scorer)
        backend.evaluate_choice(premise, "Route", {"shipping": "Logistics", "billing": "Payments"})

        # Initial 2 calls + 4 consensus calls = 6 calls
        assert len(calls) == 6, f"Expected 6 calls for {lang}, got {len(calls)}"
        consensus_calls = calls[2:]
        assert any(exp_tpl1 in c for c in consensus_calls), f"Missing {exp_tpl1} in {consensus_calls}"
        assert any(exp_tpl2 in c for c in consensus_calls), f"Missing {exp_tpl2} in {consensus_calls}"


def test_consensus_with_none_and_empty_option_descriptions() -> None:
    """Verify consensus operates safely without exceptions when descriptions are None or empty."""
    calls: list[str] = []

    def mock_narrow_scorer(state: str, inst: str, hyp: str) -> float:
        calls.append(hyp)
        if "opt_a" in hyp:
            return 1.05
        return 1.00

    backend = FastEncoderBackend(logit_scorer=mock_narrow_scorer)
    res = backend.evaluate_choice(
        "Cet état correspond à une demande urgente.",
        "Classer",
        {"opt_a": None, "opt_b": ""},
    )
    assert res["choice"] == "opt_a"
    assert len(calls) == 6  # 2 initial + 4 consensus
    assert any("opt_a: opt_a" in c for c in calls)


def test_consensus_resolves_ambiguous_decision_in_french() -> None:
    """Verify that multi-template consensus breaks a tie / flips margin when secondary templates favor the true label."""
    def decisive_consensus_scorer(state: str, inst: str, hyp: str) -> float:
        # Primary template: billing is slightly ahead (1.05 vs 1.00, margin 0.05)
        if hyp.startswith("Il s'agit de"):
            return 1.05 if "billing" in hyp else 1.00
        # Secondary consensus template: shipping is strongly ahead
        if hyp.startswith("La classification correcte est"):
            return 0.50 if "billing" in hyp else 2.50
        # Tertiary consensus template: shipping is strongly ahead
        if hyp.startswith("Cet état correspond à"):
            return 0.50 if "billing" in hyp else 2.50
        return 0.0

    backend = FastEncoderBackend(logit_scorer=decisive_consensus_scorer)
    res = backend.evaluate_choice(
        "Mon colis est en retard et je n'ai toujours pas reçu la livraison.",
        "Classer",
        {"billing": "Facturation", "shipping": "Livraison de colis"},
    )
    # Consensus average for billing: (1.05 + 0.50 + 0.50) / 3 = 0.683
    # Consensus average for shipping: (1.00 + 2.50 + 2.50) / 3 = 2.000
    assert res["choice"] == "shipping"
    assert res["probabilities"]["shipping"] > res["probabilities"]["billing"]


# ===========================================================================
# 4. Localized Negative Anchor Hypotheses in evaluate_multi
# ===========================================================================


def test_negative_anchor_quadrilingual_content_and_dispatch() -> None:
    """Verify that evaluate_multi dispatches the exact localized negative anchor for EN, FR, ES, PT."""
    expected_negative_anchors = {
        "en": "It is none: Unrelated, not applicable, false, absent.",
        "fr": "Il s'agit de aucun: Non lié, non applicable, faux, absent.",
        "es": "Se trata de ninguno: No relacionado, no aplicable, falso, ausente.",
        "pt": "Trata-se de nenhum: Não relacionado, não aplicável, falso, ausente.",
    }

    for lang, exp_anchor in expected_negative_anchors.items():
        assert _NEGATIVE_HYPOTHESES[lang] == exp_anchor

        calls: list[str] = []

        def mock_scorer(state: str, inst: str, hyp: str) -> float:
            calls.append(hyp)
            return 0.20

        backend = FastEncoderBackend(logit_scorer=mock_scorer)
        premises = {
            "en": "Please check my account details.",
            "fr": "Veuillez vérifier les détails de mon compte.",
            "es": "Por favor verifique los detalles de mi cuenta.",
            "pt": "Por favor verifique os detalhes da minha conta.",
        }
        backend.evaluate_multi(premises[lang], "Evaluate", {"opt1": "Desc 1"})
        assert calls[0] == exp_anchor, f"Expected negative anchor {exp_anchor!r}, got {calls[0]!r}"


def test_negative_anchor_neural_suppression_of_irrelevant_premises() -> None:
    """Verify calibrated neural NLI scorer uses negative anchors to suppress completely irrelevant premises."""
    def mock_neural_nli_scorer(state: str, inst: str, hyp: str) -> float:
        s_lower = state.lower()
        h_lower = hyp.lower()
        # Irrelevant text matches negative anchor, contradicts positive support categories
        if any(term in s_lower for term in ("recette", "gâteau", "pronóstico", "futebol", "botanical")):
            if any(neg in h_lower for neg in ("aucun", "ninguno", "nenhum", "none")):
                return 2.50  # Strong entailment of negative anchor
            return -2.50     # Strong contradiction of support categories
        return 0.0

    backend = FastEncoderBackend(logit_scorer=mock_neural_nli_scorer)
    options = {
        "billing": "Facturation, prélèvements, remboursements et cartes bancaires",
        "shipping": "Livraison, colis, suivi de transport et douane",
        "technical": "Support technique, pannes de serveur, bugs et erreurs",
    }

    irrelevant_premises = [
        ("fr", "La recette de cuisine pour le gâteau au chocolat nécessite de la farine et du sucre."),
        ("es", "El pronóstico del tiempo para este fin de semana en las montañas anuncia lluvia."),
        ("pt", "O resultado da partida de futebol do campeonato terminou em empate ontem à noite."),
        ("en", "The botanical classification of flowering plants in the Amazon rain forest."),
    ]

    for lang, premise in irrelevant_premises:
        res = backend.evaluate_multi(premise, "Classify", options, threshold=0.50)
        assert res["labels"] == [], f"Irrelevant {lang} premise produced false positive labels: {res['labels']}"
        assert all(p < 0.05 for p in res["probabilities"].values())
        assert res["confidence"] > 0.90


def test_negative_anchor_neural_suppression_under_explicit_rejections() -> None:
    """Verify that neural NLI scorer elevates negative anchors under explicit rejections and clears labels."""
    def mock_rejection_nli_scorer(state: str, inst: str, hyp: str) -> float:
        s_lower = state.lower()
        h_lower = hyp.lower()
        if any(w in s_lower for w in ("rien de tout cela", "ninguno de estos", "nenhum desses", "none of the above")):
            if any(neg in h_lower for neg in ("aucun", "ninguno", "nenhum", "none")):
                return 3.20  # Strong entailment of negative anchor
            return -3.00     # Strong contradiction
        return 0.0

    backend = FastEncoderBackend(logit_scorer=mock_rejection_nli_scorer)
    rejection_premises = [
        ("fr", "Rien de tout cela, aucun problème de facture ou de colis ou de bug technique."),
        ("es", "Ninguno de estos, no tengo problemas con cobros, envíos ni errores de sistema."),
        ("pt", "Nenhum desses, não se trata de pagamento, entrega ou falha técnica no sistema."),
        ("en", "None of the above, this is completely unrelated to payments, delivery, or tech."),
    ]

    categories = {
        "billing": "Billing and refunds",
        "shipping": "Shipping and delivery",
        "technical": "Technical bug reports",
    }

    for lang, premise in rejection_premises:
        res = backend.evaluate_multi(premise, "Classify labels", categories, threshold=0.50)
        assert res["labels"] == [], f"Explicit rejection for {lang} returned non-empty labels: {res['labels']}"
        assert all(p < 0.01 for p in res["probabilities"].values())


def test_multi_label_joint_activation_quadrilingual_with_calibrated_scorer() -> None:
    """Verify composite premises activating two categories simultaneously succeed across languages."""
    def mock_dual_intent_scorer(state: str, inst: str, hyp: str) -> float:
        s_lower = state.lower()
        h_lower = hyp.lower()
        is_billing = any(w in s_lower for w in ("débité", "prélevé", "cobraron", "cobrado", "charged"))
        is_shipping = any(w in s_lower for w in ("douane", "colis", "paquete", "encomenda", "customs", "delivery"))

        if "aucun" in h_lower or "ninguno" in h_lower or "nenhum" in h_lower or "none" in h_lower:
            return -2.00  # Negative anchor contradicted because valid issues exist

        score = -1.50
        if is_billing and ("billing" in h_lower or "factur" in h_lower or "fatur" in h_lower):
            score = 3.20
        elif is_shipping and ("shipping" in h_lower or "livraison" in h_lower or "envío" in h_lower or "envio" in h_lower):
            score = 3.10
        return score

    backend = FastEncoderBackend(logit_scorer=mock_dual_intent_scorer)
    dual_intent_cases = [
        ("fr", "J'ai été prélevé deux fois pour ma commande et mon colis est bloqué en douane.", ["billing", "shipping"]),
        ("es", "Me cobraron dos veces por el pedido y además mi paquete está retenido en la aduana.", ["billing", "shipping"]),
        ("pt", "Fui cobrado duas vezes pelo pedido e além disso minha encomenda está parada na alfândega.", ["billing", "shipping"]),
        ("en", "I was charged twice on my credit card and my package is delayed in delivery.", ["billing", "shipping"]),
    ]

    categories = {
        "billing": "Billing, refunds, double charges, invoices, payments",
        "shipping": "Shipping logistics, parcels, customs delay, delivery",
        "technical": "Technical crashes, database 500 error, server outage",
    }

    for lang, premise, expected_active in dual_intent_cases:
        res = backend.evaluate_multi(premise, "Select all applicable departments", categories, threshold=0.50)
        assert set(res["labels"]) == set(expected_active), (
            f"Expected {expected_active} in {lang}, got {res['labels']}"
        )
        assert "technical" not in res["labels"]


# ===========================================================================
# 5. High-Stress Boundary Conditions for End-to-End Decision Backends
# ===========================================================================


def test_hierarchical_tournament_with_localized_templates_large_option_set() -> None:
    """Verify StartLux tournament (>25 options) routes accurately with localized templates in French."""
    backend = FastEncoderBackend()
    options: dict[str, str] = {
        f"dept_{i}": f"Service support numéro {i} pour demandes générales"
        for i in range(50)
    }
    options["dept_special"] = "Service spécialisé pour remboursement urgent de prélèvement bancaire"

    state = "Je demande un remboursement immédiat pour un double prélèvement bancaire."
    res = backend.evaluate_choice(state, "Aiguiller vers le bon service", options)

    assert res["choice"] == "dept_special"
    assert len(res["probabilities"]) == 51  # 50 general + 1 special
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-5)
    assert 0.0 <= res["confidence"] <= 1.0


def test_ordinal_score_quadrilingual_with_localized_hypotheses() -> None:
    """Verify ordinal scoring across EN, FR, ES, PT with localized level descriptions."""
    backend = FastEncoderBackend()

    levels_fr = [
        "Très insatisfait, service inacceptable",
        "Insatisfait, plusieurs problèmes rencontrés",
        "Neutre, service moyen",
        "Satisfait, bonne expérience globale",
        "Très satisfait, service parfait et rapide",
    ]

    # Clearly positive review in French
    res_pos = backend.evaluate_score(
        "Expérience absolument formidable, service client parfait et très rapide!",
        "Évaluer la satisfaction client",
        levels_fr,
    )
    assert res_pos["score"] >= 2.5, f"Expected positive score >= 2.5, got {res_pos['score']}"
    assert 0.0 <= res_pos["confidence"] <= 1.0
    assert np.isclose(sum(res_pos["probabilities"].values()), 1.0, atol=1e-5)

    # Clearly negative review in Spanish
    levels_es = [
        "Muy insatisfecho, pésimo servicio",
        "Insatisfecho, problemas frecuentes",
        "Neutro, regular",
        "Satisfecho, buena atención",
        "Muy satisfecho, excelente servicio",
    ]
    res_neg = backend.evaluate_score(
        "Pésimo servicio, una estafa completa, nadie responde y perdí mi dinero.",
        "Evaluar satisfacción del cliente",
        levels_es,
    )
    assert res_neg["score"] <= 1.5, f"Expected negative score <= 1.5, got {res_neg['score']}"


def test_rubric_conflict_overrides_deceptive_keys_in_french() -> None:
    """HYP-006: Verify definition-dominant rubric binding in French when option keys are deceptive."""
    backend = FastEncoderBackend(rubric_dominance=True)
    deceptive_options = {
        # Deceptive key: key says 'livraison' (shipping), but definition is strictly billing
        "livraison": "Facturation, prélèvement bancaire, remboursement et facture d'abonnement",
        # Deceptive key: key says 'facturation', but definition is strictly shipping
        "facturation": "Livraison physique, colis bloqué, transporteur et numéro de suivi",
    }

    state = "J'ai un prélèvement injustifié sur ma carte et je veux mon remboursement."
    res = backend.evaluate_choice(state, "Aiguiller la demande", deceptive_options)

    # Because definition dominates, 'livraison' (whose definition is billing) must be selected!
    assert res["choice"] == "livraison"
    assert res["x_rubric_conflict_detected"] is True
