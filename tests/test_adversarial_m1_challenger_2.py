"""Adversarial Challenge Test Harness for Milestone 1 (Strategy 1) - Challenger 2.

Empirically challenges:
1. Negative concord in Spanish and Portuguese in FastEncoderBackend.evaluate_noul.
2. Double negation cancellation to affirmative across localized phrasing variations.
3. Mixed-language, Spanglish, Portuñol, and English loanword tickets in routing and polarity.
4. Robustness of detect_language_fast across 112 tricky multilingual short sentences.
"""

from __future__ import annotations

import pytest
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.lexicons.normalizer import detect_language_fast

ROUTING_OPTIONS = {
    "billing": "Billing, invoices, charges, double debits, subscription pricing, refunds and fee disputes",
    "shipping": "Shipping logistics, parcel tracking, courier delays, missing orders and package delivery",
    "tech": "Technical support, bug reports, application crashes, server outages and login failures",
}
INSTRUCTIONS = "Route this customer inquiry to the appropriate support department."


# ===========================================================================
# 1. Negative Concord & Polarity in FastEncoderBackend.evaluate_noul
# ===========================================================================

def test_spanish_negative_concord_premise_hypothesis_entailment() -> None:
    """Challenge Spanish negative concord in NLI entailment.

    In Spanish, negative concord ('no quiero nada', 'nunca llegó ningún paquete')
    is standard negative grammar.
    When premise has negative concord and hypothesis is also negative,
    the relationship is ENTAILMENT (truth probability should be high, >0.50).
    """
    backend = FastEncoderBackend()

    # Case 1: 'no ... nada' premise with 'no ...' hypothesis
    res1 = backend.evaluate_noul(
        "Premise: No quiero nada del paquete.\nHypothesis: El cliente no quiere el paquete."
    )
    # Case 2: 'nunca llegó ningún paquete' premise with 'el paquete no llegó' hypothesis
    res2 = backend.evaluate_noul(
        "Premise: Nunca llegó ningún paquete a mi casa.\nHypothesis: El paquete no llegó a la casa."
    )

    assert res1["noul"] > 0.50, f"Expected entailment (>0.50) for 'no quiero nada', got {res1['noul']:.4f}"
    assert res2["noul"] > 0.50, f"Expected entailment (>0.50) for 'nunca llegó ningún', got {res2['noul']:.4f}"


def test_spanish_negative_concord_contradiction() -> None:
    """Challenge Spanish negative concord vs affirmative hypothesis (must be CONTRADICTION, <0.35)."""
    backend = FastEncoderBackend()

    res1 = backend.evaluate_noul(
        "Premise: No quiero nada del paquete.\nHypothesis: El cliente quiere el paquete."
    )
    res2 = backend.evaluate_noul(
        "Premise: Nunca llegó ningún paquete a mi casa.\nHypothesis: El paquete llegó a la casa."
    )
    assert res1["noul"] < 0.35, f"Expected contradiction for res1, got {res1['noul']}"
    assert res2["noul"] < 0.35, f"Expected contradiction for res2, got {res2['noul']}"


def test_portuguese_negative_concord_premise_hypothesis_entailment() -> None:
    """Challenge Portuguese negative concord in NLI entailment.

    In Portuguese, negative concord ('não recebi nada', 'nunca foi entregue nenhum pedido')
    is standard negative grammar.
    When premise has negative concord and hypothesis is also negative,
    the relationship is ENTAILMENT (truth probability should be high, >0.50).
    """
    backend = FastEncoderBackend()

    # Case 1: 'não recebi nada' premise with 'não recebeu o pacote' hypothesis
    res1 = backend.evaluate_noul(
        "Premise: Não recebi nada no meu endereço.\nHypothesis: O cliente não recebeu o pacote no endereço."
    )
    # Case 2: 'nunca foi entregue nenhum pedido' premise with 'o pedido não foi entregue' hypothesis
    res2 = backend.evaluate_noul(
        "Premise: Nunca foi entregue nenhum pedido.\nHypothesis: O pedido não foi entregue."
    )

    assert res1["noul"] > 0.50, f"Expected entailment (>0.50) for 'não recebi nada', got {res1['noul']:.4f}"
    assert res2["noul"] > 0.50, f"Expected entailment (>0.50) for 'nunca foi entregue nenhum', got {res2['noul']:.4f}"


def test_portuguese_negative_concord_contradiction() -> None:
    """Challenge Portuguese negative concord vs affirmative hypothesis (must be CONTRADICTION, <0.35)."""
    backend = FastEncoderBackend()

    res1 = backend.evaluate_noul(
        "Premise: Não recebi nada no meu endereço.\nHypothesis: O cliente recebeu o pacote no endereço."
    )
    res2 = backend.evaluate_noul(
        "Premise: Nunca foi entregue nenhum pedido.\nHypothesis: O pedido foi entregue."
    )
    assert res1["noul"] < 0.35, f"Expected contradiction for res1, got {res1['noul']}"
    assert res2["noul"] < 0.35, f"Expected contradiction for res2, got {res2['noul']}"


# ===========================================================================
# 2. Double Negation Cancellation to Affirmative
# ===========================================================================

def test_double_negation_cancellation_working_cases() -> None:
    """Verify double negation phrases that currently succeed in canceling to affirmative truth (>0.50)."""
    backend = FastEncoderBackend()

    res_es1 = backend.evaluate_noul(
        "Premise: No es imposible recuperar el acceso a la cuenta.\nHypothesis: Es posible recuperar el acceso."
    )
    assert res_es1["noul"] > 0.60, f"Expected affirmative (>0.60), got {res_es1['noul']:.4f}"

    res_pt1 = backend.evaluate_noul(
        "Premise: Não é impossível recuperar o acesso à conta.\nHypothesis: É possível recuperar o acesso à conta."
    )
    assert res_pt1["noul"] > 0.60, f"Expected affirmative (>0.60), got {res_pt1['noul']:.4f}"


def test_portuguese_double_negation_nao_e_improvavel() -> None:
    """Challenge Portuguese 'não é improvável' canceling to affirmative."""
    backend = FastEncoderBackend()
    res = backend.evaluate_noul(
        "Premise: Não é improvável recuperar a conta.\nHypothesis: É provável recuperar a conta."
    )
    assert res["noul"] > 0.50, f"Expected affirmative (>0.50), got {res['noul']:.4f}"


def test_spanish_double_negation_accented_phrase() -> None:
    """Challenge Spanish accented double negation phrase 'no está prohibido'."""
    backend = FastEncoderBackend()
    res = backend.evaluate_noul(
        "Premise: El acceso no está prohibido para los usuarios.\nHypothesis: El acceso está permitido para los usuarios."
    )
    assert res["noul"] > 0.50, f"Expected affirmative (>0.50), got {res['noul']:.4f}"


# ===========================================================================
# 3. Mixed-Language / Loanword Tickets (Spanglish, Portuñol, English terms)
# ===========================================================================

def test_mixed_language_spanglish_and_portunol_routing() -> None:
    """Challenge ticket routing on Spanglish, Portuñol, and English loanword tickets."""
    backend = FastEncoderBackend()

    mixed_tickets = [
        # Spanglish Billing
        ("Hice un refund pero el claim sigue pending en mi credit card account.", "billing"),
        ("Tengo un charge duplicado en el invoice de este mes y quiero money back.", "billing"),
        # Spanglish Shipping
        ("El delivery driver me dejó el parcel completamente broken en el lobby.", "shipping"),
        ("Mi tracking number no actualiza el shipping status del paquete.", "shipping"),
        # Spanglish Tech
        ("Tengo un bug en el login y la mobile app crashea con 500 fatal error.", "tech"),
        ("El backend server está down y la API connection tira timeout continuo.", "tech"),
        # Portuñol Billing
        ("Preciso de un reembolso porque houve cobro doble no cartão de crédito.", "billing"),
        ("A fatura deste mês veio com cobro indebido e quero estorno rápido.", "billing"),
        # Portuñol Shipping
        ("El pacote no chegou no endereço e o rastreamento está atrasado.", "shipping"),
        ("A encomenda foi entregue com a caixa rota e o producto danificado.", "shipping"),
        # Portuñol Tech
        ("O servidor está caído con erro 500 e não consigo hacer login na conta.", "tech"),
        ("A aplicação trava con un fallo crítico na base de dados.", "tech"),
        # Heavy Loanwords in Spanish/Portuguese
        ("El payment gateway arrojó un error de checkout y me duplicó la suscripción.", "billing"),
        ("El courier reportó failed delivery pero nadie tocó el timbre de la casa.", "shipping"),
        ("O estafeta fez drop-off da encomenda mas o pacote está missing.", "shipping"),
    ]

    correct = 0
    total = len(mixed_tickets)
    for ticket, expected in mixed_tickets:
        res = backend.evaluate_choice(ticket, INSTRUCTIONS, ROUTING_OPTIONS)
        if res["choice"] == expected:
            correct += 1

    acc = correct / total
    assert acc >= 0.85, f"Mixed-language routing accuracy {acc * 100:.1f}% fell below 85%"


# ===========================================================================
# 4. Language Detection Robustness (112 Tricky Multilingual Short Sentences)
# ===========================================================================

TRICKY_MULTILINGUAL_SENTENCES = [
    # --- Short English (25 sentences) ---
    ("en", "I need help with my bill."),
    ("en", "Where is my parcel?"),
    ("en", "The server is completely down."),
    ("en", "Please cancel this invoice immediately."),
    ("en", "My order has not arrived yet."),
    ("en", "Login failed with error 500."),
    ("en", "Can you refund my money?"),
    ("en", "I did not receive the delivery."),
    ("en", "The application crashed on startup."),
    ("en", "Why was I charged twice?"),
    ("en", "We have a severe outage."),
    ("en", "There is no package at my door."),
    ("en", "Payment failed on checkout."),
    ("en", "Please help me with this charge."),
    ("en", "Delivery was delayed by courier."),
    ("en", "The database connection was refused."),
    ("en", "I want to return this product."),
    ("en", "Could you check the order status?"),
    ("en", "My account is locked out."),
    ("en", "Refund was not processed."),
    ("en", "Is there an update on delivery?"),
    ("en", "Our API token is invalid."),
    ("en", "You charged my card without authorization."),
    ("en", "Package arrived broken and torn."),
    ("en", "Application is frozen on welcome screen."),

    # --- Short French (25 sentences) ---
    ("fr", "J'ai besoin d'aide pour ma facture."),
    ("fr", "Où est mon colis?"),
    ("fr", "Le serveur est complètement en panne."),
    ("fr", "Veuillez annuler cette facture immédiatement."),
    ("fr", "Ma commande n'est pas encore arrivée."),
    ("fr", "La connexion a échoué avec une erreur 500."),
    ("fr", "Pouvez-vous me rembourser mon argent?"),
    ("fr", "Je n'ai pas reçu la livraison."),
    ("fr", "L'application a planté au démarrage."),
    ("fr", "Pourquoi ai-je été débité deux fois?"),
    ("fr", "Nous avons une panne grave."),
    ("fr", "Il n'y a pas de colis devant ma porte."),
    ("fr", "Le paiement a échoué lors de la commande."),
    ("fr", "Aidez-moi avec ce prélèvement."),
    ("fr", "La livraison est en retard avec le transporteur."),
    ("fr", "La connexion à la base de données est coupée."),
    ("fr", "Je souhaite retourner ce produit."),
    ("fr", "Pouvez-vous vérifier le statut de la commande?"),
    ("fr", "Mon compte est bloqué."),
    ("fr", "Le remboursement n'a pas été effectué."),
    ("fr", "Y a-t-il une mise à jour sur la livraison?"),
    ("fr", "Notre jeton d'accès est invalide."),
    ("fr", "Vous avez débité ma carte sans autorisation."),
    ("fr", "Le colis est arrivé cassé et déchiré."),
    ("fr", "L'application est figée sur l'écran d'accueil."),

    # --- Short Spanish (25 sentences) ---
    ("es", "Necesito ayuda con mi factura."),
    ("es", "¿Dónde está mi paquete?"),
    ("es", "El servidor está completamente caído."),
    ("es", "Por favor cancelen esta factura inmediatamente."),
    ("es", "Mi pedido no ha llegado todavía."),
    ("es", "El inicio de sesión falló con error 500."),
    ("es", "¿Pueden devolverme mi dinero?"),
    ("es", "No he recibido la entrega."),
    ("es", "La aplicación se cerró al iniciar."),
    ("es", "¿Por qué me cobraron dos veces?"),
    ("es", "Tenemos una avería grave en el sistema."),
    ("es", "No hay ningún paquete en mi puerta."),
    ("es", "El pago falló en la pasarela."),
    ("es", "Ayúdenme con este cobro indebido."),
    ("es", "La entrega tiene retraso con el mensajero."),
    ("es", "La conexión a la base de datos fue rechazada."),
    ("es", "Quiero devolver este producto dañado."),
    ("es", "¿Pueden comprobar el estado de mi envío?"),
    ("es", "Mi cuenta de usuario está bloqueada."),
    ("es", "El reembolso no fue procesado aún."),
    ("es", "¿Hay alguna novedad con mi entrega?"),
    ("es", "Nuestro token de API no es válido."),
    ("es", "Me cobraron la tarjeta sin autorización."),
    ("es", "El paquete llegó roto y destrozado."),
    ("es", "La aplicación web está congelada."),

    # --- Short Portuguese (25 sentences) ---
    ("pt", "Preciso de ajuda com a minha fatura."),
    ("pt", "Onde está a minha encomenda?"),
    ("pt", "O servidor está completamente fora do ar."),
    ("pt", "Por favor cancelem esta fatura imediatamente."),
    ("pt", "O meu pedido ainda não chegou."),
    ("pt", "A autenticação falhou com erro 500."),
    ("pt", "Podem estornar o meu dinheiro?"),
    ("pt", "Não recebi a entrega no meu endereço."),
    ("pt", "O aplicativo travou ao iniciar."),
    ("pt", "Por que fui cobrado duas vezes no cartão?"),
    ("pt", "Estamos com uma pane técnica grave."),
    ("pt", "Não há nenhum pacote na minha porta."),
    ("pt", "O pagamento falhou na finalização da compra."),
    ("pt", "Ajudem-me com esta cobrança indevida."),
    ("pt", "A entrega está atrasada nos correios."),
    ("pt", "A conexão com o banco de dados falhou."),
    ("pt", "Quero devolver este produto danificado."),
    ("pt", "Podem verificar o status do meu rastreamento?"),
    ("pt", "A minha conta está bloqueada no sistema."),
    ("pt", "O estorno não foi processado ainda."),
    ("pt", "Há alguma novidade sobre a encomenda?"),
    ("pt", "O nosso token de acesso é inválido."),
    ("pt", "Cobraram o meu cartão sem autorização."),
    ("pt", "O pacote chegou quebrado e rasgado."),
    ("pt", "O aplicativo móvel está congelado na tela."),

    # --- Extreme Cognates & Tricky Cross-Lingual Sentences (12 sentences) ---
    ("pt", "Não, obrigado."),
    ("es", "No, gracias."),
    ("fr", "Non, merci."),
    ("en", "No, thanks."),
    ("pt", "Muito obrigado pela atenção."),
    ("es", "Muchas gracias por la atención."),
    ("fr", "Merci beaucoup pour votre attention."),
    ("en", "Thank you very much for your attention."),
    ("pt", "Sim, está correto e confirmado."),
    ("es", "Sí, está correcto y confirmado."),
    ("fr", "Oui, c'est correct et confirmé."),
    ("en", "Yes, it is correct and confirmed."),
]


def test_detect_language_fast_112_tricky_sentences_benchmark() -> None:
    """Benchmark detect_language_fast on 112 tricky short sentences across EN, FR, ES, PT."""
    total = len(TRICKY_MULTILINGUAL_SENTENCES)
    correct = 0

    for expected_lang, sentence in TRICKY_MULTILINGUAL_SENTENCES:
        predicted = detect_language_fast(sentence)
        if predicted == expected_lang:
            correct += 1

    overall_acc = correct / float(total)
    assert overall_acc >= 0.90, (
        f"Language detection accuracy {overall_acc * 100:.1f}% fell below 90% target ({correct}/{total})"
    )
