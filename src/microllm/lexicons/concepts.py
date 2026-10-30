"""Multilingual concept clusters for System-One zero-shot CPU scoring (Gear 1).

Covers 10 concept clusters across English (EN), French (FR), Spanish (ES),
Portuguese (PT), and Korean (KO), accounting for Peninsular/Latin-American Spanish
and European/Brazilian Portuguese dialectal variants, with both accented and unaccented forms.
"""

from __future__ import annotations

# Strict word-boundary security keywords that require whole-token matches
STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS: frozenset[str] = frozenset(
    {"format", "table", "system", "credit", "private"}
)

CONCEPT_CLUSTERS: dict[str, set[str]] = {
    # Slot 0: Billing & Finance
    "billing_finance": {
        # English
        "bill", "billed", "billing", "invoice", "invoices", "charge", "charged",
        "double-charged", "refund", "payment", "payments", "subscription", "card",
        "price", "pricing", "fee", "overcharged", "receipt", "receipts",
        # French
        "facture", "factures", "facturation", "remboursement", "rembourser",
        "remboursé", "remboursee", "paiement", "paiements", "payer", "payé",
        "paye", "abonnement", "abonnements", "carte", "tarif", "tarifs", "prix",
        "frais", "surcharge", "prelevement", "prélèvement", "virement", "devis",
        "avoir", "recu", "reçu",
        # Korean
        "결제", "환불", "청구",
        # Spanish (Peninsular & Latin-American, accented & unaccented)
        "factura", "facturas", "facturación", "facturacion", "facturar", "facturado",
        "facturada", "reembolso", "reembolsos", "reembolsar", "reembolsado",
        "reembolsada", "cobro", "cobros", "cobrado", "cobrada", "cobrar",
        "sobrecargo", "sobrecargos", "sobracargo", "pago", "pagos", "pagar",
        "pagado", "pagada", "suscripción", "suscripcion", "suscripciones",
        "tarjeta", "tarjetas", "precio", "precios", "tarifa", "tarifas", "cuota",
        "cuotas", "comisión", "comision", "comisiones", "recibo", "recibos",
        "cuenta", "cuentas", "saldo", "devolución", "devolucion", "devoluciones",
        "duplicado", "duplicada",
        # Portuguese (European & Brazilian, accented & unaccented)
        "fatura", "faturas", "faturamento", "faturar", "faturado", "faturada",
        "factura", "facturas", "facturação", "facturacao", "facturar",
        "cobrança", "cobranca", "cobranças", "cobrancas", "cobrado", "cobrada",
        "cobrar", "sobretaxa", "pagamento", "pagamentos", "assinatura", "assinaturas",
        "cartão", "cartao", "cartões", "cartoes", "preço", "preco", "preços", "precos",
        "taxa", "taxas", "estorno", "estornos", "extrato", "boleto", "boletos", "pix",
        # German (cross-lingual robustness)
        "rechnung", "ruckerstattung", "rückerstattung",
    },

    # Slot 1: Technical Support
    "technical_support": {
        # English
        "tech", "technical", "bug", "bugs", "crash", "crashes", "error", "errors",
        "outage", "server", "login", "password", "api", "slow", "broken", "down",
        "exception", "stacktrace", "issue", "failure", "freeze",
        # French
        "panne", "pannes", "erreur", "erreurs", "plantage", "planter", "plante",
        "serveur", "serveurs", "connexion", "motdepasse", "identifiant", "lent",
        "lenteur", "bloque", "bloqué", "bugue", "dysfonctionnement", "deconnexion",
        "déconnexion", "incident", "incidents", "assistance", "support",
        # Korean
        "기술", "지원",
        # Spanish (Peninsular & Latin-American, accented & unaccented)
        "técnico", "tecnico", "técnica", "tecnica", "soporte", "fallo", "fallos",
        "falla", "fallas", "fallar", "fallando", "avería", "averia", "averías", "averias",
        "error", "errores", "caída", "caida", "caído", "caido", "caídas", "caidas",
        "bloqueo", "bloqueado", "bloqueada", "bloquear", "congelado", "congelada",
        "cuelgue", "colgado", "pantallazo", "servidor", "servidores", "contraseña",
        "contrasena", "clave", "acceso", "sesión", "sesion", "lento", "lenta",
        "lentitud", "demora", "roto", "rota", "estropeado", "descompuesto",
        "descompuesta", "incidencia", "incidencias", "excepción", "excepcion",
        "desconexión", "desconexion",
        # Portuguese (European & Brazilian, accented & unaccented)
        "técnico", "tecnico", "técnica", "tecnica", "suporte", "falha", "falhas",
        "defeito", "defeitos", "avaria", "avarias", "pane", "panes", "erro", "erros",
        "queda", "quedas", "caiu", "travamento", "travamentos", "travou", "travado",
        "travada", "bloqueio", "bloqueado", "bloqueada", "bloquear", "servidor",
        "servidores", "senha", "senhas", "acesso", "autenticação", "autenticacao",
        "lento", "lenta", "lentidão", "lentidao", "quebrado", "quebrada",
        "estragado", "estragada", "bugado", "bugada", "incidente", "incidentes",
        "instabilidade", "desconexão", "desconexao",
    },

    # Slot 2: Shipping & Logistics
    "shipping_logistics": {
        # English
        "ship", "shipping", "delivery", "parcel", "package", "courier", "late",
        "delayed", "lost", "tracking", "order", "arrived", "arrival", "mailbox",
        "shown", "showed", "transit", "postage", "damaged", "broken", "box", "torn",
        # French
        "livraison", "livraisons", "colis", "livrer", "livre", "livré", "colissimo",
        "transporteur", "expedition", "expédition", "expedie", "expédié", "retard",
        "retarde", "retardé", "perdu", "perdue", "suivi", "commande", "commandes",
        "arrive", "arrivé", "boite", "boîte", "reception", "réception",
        "endommagé", "endommage", "cassé", "casse", "abîmé", "abime",
        # Spanish (Peninsular & Latin-American, accented & unaccented)
        "envío", "envio", "envíos", "envios", "enviar", "enviado", "enviada",
        "entrega", "entregas", "entregar", "entregado", "entregada", "paquete",
        "paquetes", "paquetería", "paqueteria", "encomienda", "encomiendas",
        "despacho", "despachos", "despachado", "despachada", "mensajería",
        "mensajeria", "mensajero", "repartidor", "correo", "correos", "tarde",
        "retraso", "retrasos", "retrasado", "retrasada", "demorado", "demorada",
        "perdido", "perdida", "extraviado", "extraviada", "seguimiento", "rastreo",
        "localizador", "pedido", "pedidos", "orden", "ordenes", "órdenes",
        "llegada", "llegó", "llego", "llegado", "recibido", "recibida",
        "buzón", "buzon", "dirección", "direccion", "domicilio",
        "dañado", "dañada", "roto", "rota", "destrozado", "destrozada", "caja",
        # Portuguese (European & Brazilian, accented & unaccented)
        "envio", "envios", "enviar", "enviado", "enviada", "entrega", "entregas",
        "entregar", "entregue", "entregues", "pacote", "pacotes", "encomenda",
        "encomendas", "despacho", "despachos", "despachado", "despachada",
        "transportadora", "transportador", "estafeta", "estafetas", "entregador",
        "entregadores", "correio", "correios", "sedex", "atraso", "atrasos",
        "atrasado", "atrasada", "perdido", "perdida", "extraviado", "extraviada",
        "rastreamento", "rastreamentos", "rastreio", "rastreios", "pedido",
        "pedidos", "chegada", "chegou", "recebido", "recebida", "caixa",
        "endereço", "endereco", "morada", "domicílio", "domicilio",
        "danificado", "danificada", "quebrado", "quebrada", "violado", "violada",
    },

    # Slot 3: High Urgency
    "urgency_high": {
        # English
        "urgent", "immediately", "immediate", "today", "now", "critical", "blocking",
        "outage", "furious", "angry", "cancel", "twice", "still", "escalate",
        "emergency", "severe", "high", "asap",
        # French
        "immédiat", "immediat", "immédiate", "immediate", "urgent", "urgente",
        "urgence", "urgences", "immediatement", "immédiatement", "maintenant",
        "critique", "bloquant", "bloquante", "furieux", "furieuse", "enragé",
        "annuler", "annulation", "escalader", "grave", "inadmissible", "inacceptable",
        "alerte", "danger",
        # Spanish (Peninsular & Latin-American, accented & unaccented)
        "urgente", "urgentes", "urgencia", "urgencias", "inmediatamente",
        "inmediato", "inmediata", "hoy", "ahora", "ya", "enseguida", "pronto",
        "crítico", "critico", "crítica", "critica", "bloqueante", "furia",
        "furioso", "furiosa", "enfadado", "enfadada", "enojado", "enojada",
        "cancelar", "cancelación", "cancelacion", "anular", "escalar",
        "emergencia", "emergencias", "grave", "graves", "severo", "severa",
        "inadmisible", "inaceptable", "prioridad", "alerta", "peligro",
        # Portuguese (European & Brazilian, accented & unaccented)
        "urgente", "urgentes", "urgência", "urgencia", "imediatamente",
        "imediato", "imediata", "hoje", "agora", "já", "ja", "crítico",
        "critico", "crítica", "critica", "bloqueante", "bloqueador", "furioso",
        "furiosa", "irritado", "irritada", "revoltado", "revoltada", "cancelar",
        "cancelamento", "cancelamentos", "anular", "escalar", "emergência",
        "emergencia", "emergências", "emergencias", "grave", "graves", "severo",
        "severa", "inadmissível", "inadmissivel", "inaceitável", "inaceitavel",
        "prioridade", "alerta", "perigo",
    },

    # Slot 4: Low Urgency
    "urgency_low": {
        # English
        "calm", "low", "minor", "whenever", "question", "curious", "fine",
        "great", "happy", "satisfied", "normal", "not urgent", "routine",
        # French
        "calme", "faible", "mineur", "mineure", "question", "curieux", "curieuse",
        "satisfait", "satisfaite", "normal", "normale", "pas urgent", "sans urgence",
        "tranquille",
        # Spanish (Peninsular & Latin-American, accented & unaccented)
        "calma", "tranquilo", "tranquila", "bajo", "baja", "menor", "menores",
        "leve", "leves", "duda", "consulta", "pregunta", "curioso", "curiosa",
        "bien", "perfecto", "feliz", "satisfecho", "satisfecha", "normal",
        "no urgente", "rutina", "informativo",
        # Portuguese (European & Brazilian, accented & unaccented)
        "calmo", "calma", "tranquilo", "tranquila", "baixo", "baixa", "menor",
        "menores", "leve", "leves", "dúvida", "duvida", "consulta", "pergunta",
        "curioso", "curiosa", "bem", "ótimo", "otimo", "satisfeito", "satisfeita",
        "normal", "não urgente", "nao urgente", "rotina", "informativo",
    },

    # Slot 5: Affirmative
    "affirmative": {
        # English
        "yes", "true", "valid", "safe", "allow", "approved", "confirm", "urgent",
        "refund", "cancel", "outage", "positive",
        # French
        "oui", "vrai", "valide", "autorise", "autorisé", "approuve", "approuvé",
        "confirme", "confirmé", "exact", "positif", "correct",
        # Spanish
        "sí", "si", "verdadero", "verdadera", "válido", "valido", "válida", "valida",
        "seguro", "segura", "permitir", "permitido", "permitida", "aprobado",
        "aprobada", "confirmar", "confirmado", "confirmada", "correcto", "correcta",
        "positivo", "positiva",
        # Portuguese
        "sim", "verdadeiro", "verdadeira", "válido", "valido", "válida", "valida",
        "seguro", "segura", "permitir", "permitido", "permitida", "aprovado",
        "aprovada", "confirmar", "confirmado", "confirmada", "correto", "correta",
        "positivo", "positiva",
    },

    # Slot 6: Negative
    "negative": {
        # English
        "no", "false", "never", "not", "none", "calm", "minor", "negative",
        # French
        "non", "faux", "jamais", "rien", "aucun", "aucune", "nul", "nulle",
        "negatif", "négatif", "refuse", "refusé", "interdit",
        # Spanish
        "no", "falso", "falsa", "nunca", "jamás", "jamas", "ninguno", "ninguna",
        "ningún", "ningun", "nada", "nadie", "tampoco", "negativo", "negativa",
        "rechazado", "rechazada", "prohibido", "prohibida",
        # Portuguese
        "não", "nao", "falso", "falsa", "nunca", "jamais", "nenhum", "nenhuma",
        "nada", "ninguém", "ninguem", "tampouco", "negativo", "negativa",
        "recusado", "recusada", "proibido", "proibida",
    },

    # Slot 7: Destructive Security
    "security_destructive": {
        "rm", "-rf", "drop", "truncate", "delete", "wipe", "format disk", "destroy",
        "destructive", "production", "postgresql", "drop table", "database",
        # Multilingual additions
        "eliminar", "borrar", "destruir", "destructivo", "destructiva", "formatear disco",
        "base de datos", "apagar", "deletar", "destrutivo", "destrutiva", "formatar disco",
        "banco de dados", "supprimer", "effacer", "detruire", "détruire", "destructeur",
        "base de donnees", "base de données",
    },

    # Slot 8: Prompt Injection Security
    "security_injection": {
        "ignore", "previous", "instructions", "jailbreak", "bypass", "override",
        "system", "prompt", "hack", "injection", "disregard",
        # Multilingual additions
        "ignorar", "instrucciones previas", "inyeccion", "inyección", "saltarse",
        "anular instrucciones", "piratear", "instruções anteriores", "instrucoes anteriores",
        "injecao", "injeção", "burlar", "anular instruções", "hackear", "instructions precedentes",
        "instructions précédentes", "contourner", "piratage",
    },

    # Slot 9: PII & Credential Security
    "security_pii": {
        "password", "secret", "token", "apikey", "api_key", "ssn", "credit",
        "private", "bearer", "credentials", "leak",
        # Multilingual additions
        "contraseña", "contrasena", "clave", "secreto", "privado", "credenciales",
        "filtración", "filtracion", "fuga", "senha", "segredo", "chave secreta",
        "vazamento", "mot de passe", "prive", "privé", "identifiants", "fuite",
    },
}
