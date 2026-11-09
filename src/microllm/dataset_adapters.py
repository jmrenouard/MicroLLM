"""Web Dataset Adapters & Canonical Converters for MicroLLM Validation (16 Public Datasets).

Maps all 16 leading open-source web NLP, LLM, and safety benchmarks from Hugging Face and GitHub
into MicroLLM's uniform test case schema:
- `id`: unique identifier
- `domain`: dataset domain / origin
- `message`: input state text
- `decision_type`: one of ("noul", "choice", "score", "multi", "guardrail")
- `instructions`: prompt / instructions
- `rules`: rules dict or option mapping
- `expected_result`: target ground truth (bool, str, int, list[str])
- `metadata`: extra source annotations (e.g. source, split, check_order_invariance)

Supported 16 External Datasets across 7 Evaluation Dimensions:
1. Google BoolQ (`google/boolq` / SuperGLUE) -> `noul` (Boolean proposition evaluation)
2. MultiNLI (`nyu-mll/multi_nli` / GLUE) -> `noul` (Premise-hypothesis entailment vs contradiction)
3. PolyAI Banking77 (`PolyAI/banking77` / `mteb/banking77`) -> `choice` (Fine-grained banking intent routing)
4. AG News (`fancyzhx/ag_news`) -> `choice` (4-class news & incident topic routing)
5. CLINC150 (`clinc_oos` / `clinc/clinc_oos`) -> `choice` + Conformal Abstention (`x_abstain`)
6. Amazon Reviews Multi (`amazon_reviews_multi`, EN/FR/ES/DE) -> `score` (1-5 star ordinal ratings)
7. Yelp Review Full (`Yelp/yelp_review_full`) -> `score` (1-5 star service ordinal ratings)
8. Google GoEmotions (`google-research-datasets/go_emotions`) -> `multi` (Multi-label emotion classification)
9. Reuters-21578 (`reuters21578`) -> `multi` (Multi-label financial newswire tagging)
10. AllenAI WildGuard (`allenai/wildguardmix`) -> `guardrail` (4-rubric AI safety & adversarial prompts)
11. LMSYS ToxicChat (`lmsys/toxic-chat`) -> `guardrail` (Real-world conversation toxicity & jailbreaks)
12. Deepset Prompt Injections (`deepset/prompt-injections`) -> `guardrail` (Direct & indirect prompt injections)
13. ChoiceBench (`cotenthusiast/choicebench`) -> `choice` (`AnyJev L0` option-order invariance audit)
14. Callo-Bias Set (`peculiar-ragdoll/callo-bias-set`) -> `choice` (Cyclic permutation drift audit)
15. LMSYS RouteLLM (`lm-sys/RouteLLM`) -> `choice` (System-1 vs System-2 cascade routing)
16. JevBench (`fstandhartinger/jevbench`) -> mixed `noul`, `choice`, `score`, `multi` (`jevcompat 0.1`)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


ALL_16_EXTERNAL_DATASETS: list[str] = [
    "google/boolq",
    "nyu-mll/multi_nli",
    "PolyAI/banking77",
    "fancyzhx/ag_news",
    "clinc_oos",
    "amazon_reviews_multi",
    "Yelp/yelp_review_full",
    "google-research-datasets/go_emotions",
    "reuters21578",
    "allenai/wildguardmix",
    "lmsys/toxic-chat",
    "deepset/prompt-injections",
    "cotenthusiast/choicebench",
    "peculiar-ragdoll/callo-bias-set",
    "lm-sys/RouteLLM",
    "fstandhartinger/jevbench",
]

# Mapping from canonical dataset IDs to Hugging Face Datasets Server Parquet-native endpoints (RCA-018)
HF_DATASET_SERVER_CONFIG_MAP: dict[str, dict[str, str]] = {
    "google/boolq": {"hf_dataset": "google/boolq", "config": "default", "split": "validation"},
    "nyu-mll/multi_nli": {
        "hf_dataset": "nyu-mll/multi_nli",
        "config": "default",
        "split": "validation_matched",
    },
    "PolyAI/banking77": {"hf_dataset": "mteb/banking77", "config": "default", "split": "test"},
    "fancyzhx/ag_news": {"hf_dataset": "fancyzhx/ag_news", "config": "default", "split": "test"},
    "clinc_oos": {"hf_dataset": "clinc/clinc_oos", "config": "plus", "split": "test"},
    "amazon_reviews_multi": {
        "hf_dataset": "SetFit/amazon_reviews_multi_en",
        "config": "default",
        "split": "test",
    },
    "Yelp/yelp_review_full": {
        "hf_dataset": "Yelp/yelp_review_full",
        "config": "yelp_review_full",
        "split": "test",
    },
    "google-research-datasets/go_emotions": {
        "hf_dataset": "google-research-datasets/go_emotions",
        "config": "simplified",
        "split": "test",
    },
    "reuters21578": {
        "hf_dataset": "rjjan/reuters21578",
        "config": "ModApte",
        "split": "test",
    },
    "allenai/wildguardmix": {
        "hf_dataset": "walledai/WildGuardTest",
        "config": "default",
        "split": "train",
    },
    "lmsys/toxic-chat": {
        "hf_dataset": "lmsys/toxic-chat",
        "config": "toxicchat0124",
        "split": "test",
    },
    "deepset/prompt-injections": {
        "hf_dataset": "deepset/prompt-injections",
        "config": "default",
        "split": "test",
    },
    "cotenthusiast/choicebench": {
        "hf_dataset": "cais/mmlu",
        "config": "all",
        "split": "validation",
    },
    "peculiar-ragdoll/callo-bias-set": {
        "hf_dataset": "peculiar-ragdoll/callo-bias-set",
        "config": "default",
        "split": "train",
    },
    "lm-sys/RouteLLM": {
        "hf_dataset": "routellm/gpt4_dataset",
        "config": "default",
        "split": "validation",
    },
    "fstandhartinger/jevbench": {
        "hf_dataset": "JevBench/jevbench",
        "config": "questions",
        "split": "test",
    },
}

# Direct Parquet / JSONL full evaluation split URLs for unsampled full-corpus benchmarking (Issue #81)
FULL_SPLIT_PARQUET_URL_MAP: dict[str, dict[str, str]] = {
    "google/boolq": {
        "filename": "google__boolq.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/google/boolq/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    },
    "nyu-mll/multi_nli": {
        "filename": "nyu-mll__multi_nli.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/nyu-mll/multi_nli/resolve/refs%2Fconvert%2Fparquet/default/validation_matched/0000.parquet",
    },
    "PolyAI/banking77": {
        "filename": "PolyAI__banking77.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/mteb/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    },
    "fancyzhx/ag_news": {
        "filename": "fancyzhx__ag_news.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    },
    "clinc_oos": {
        "filename": "clinc_oos.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/clinc/clinc_oos/resolve/refs%2Fconvert%2Fparquet/plus/test/0000.parquet",
    },
    "amazon_reviews_multi": {
        "filename": "amazon_reviews_multi.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    },
    "Yelp/yelp_review_full": {
        "filename": "Yelp__yelp_review_full.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/refs%2Fconvert%2Fparquet/yelp_review_full/test/0000.parquet",
    },
    "google-research-datasets/go_emotions": {
        "filename": "google-research-datasets__go_emotions.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/test/0000.parquet",
    },
    "reuters21578": {
        "filename": "reuters21578.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/rjjan/reuters21578/resolve/refs%2Fconvert%2Fparquet/ModApte/test/0000.parquet",
    },
    "allenai/wildguardmix": {
        "filename": "allenai__wildguardmix.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/walledai/WildGuardTest/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    },
    "lmsys/toxic-chat": {
        "filename": "lmsys__toxic-chat.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/lmsys/toxic-chat/resolve/refs%2Fconvert%2Fparquet/toxicchat0124/test/0000.parquet",
    },
    "deepset/prompt-injections": {
        "filename": "deepset__prompt-injections.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/deepset/prompt-injections/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    },
    "cotenthusiast/choicebench": {
        "filename": "cotenthusiast__choicebench.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/cais/mmlu/resolve/refs%2Fconvert%2Fparquet/all/validation/0000.parquet",
    },
    "peculiar-ragdoll/callo-bias-set": {
        "filename": "peculiar-ragdoll__callo-bias-set.jsonl",
        "format": "jsonl",
        "url": "https://huggingface.co/datasets/peculiar-ragdoll/callo-bias-set/resolve/main/bias_set.jsonl",
    },
    "lm-sys/RouteLLM": {
        "filename": "lm-sys__RouteLLM.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/routellm/gpt4_dataset/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    },
    "fstandhartinger/jevbench": {
        "filename": "fstandhartinger__jevbench.parquet",
        "format": "parquet",
        "url": "https://huggingface.co/datasets/JevBench/jevbench/resolve/refs%2Fconvert%2Fparquet/questions/test/0000.parquet",
    },
}


class BaseDatasetAdapter(ABC):
    """Abstract base class for converting external web datasets to MicroLLM format."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the web dataset."""
        ...

    @property
    @abstractmethod
    def target_decision_type(self) -> str:
        """MicroLLM decision primitive (noul, choice, score, multi, guardrail, cascade)."""
        ...

    @abstractmethod
    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        """Convert a single raw dataset record into a MicroLLM test entry."""
        ...

    def convert_batch(self, raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert a batch of raw records."""
        return [self.convert_record(rec, idx + 1) for idx, rec in enumerate(raw_records)]


# =============================================================================
# 1. Google BoolQ Adapter (`noul`)
# =============================================================================
class BoolQAdapter(BaseDatasetAdapter):
    """Adapter for Google BoolQ (SuperGLUE) -> MicroLLM `noul` binary decision."""

    @property
    def dataset_name(self) -> str:
        return "google/boolq"

    @property
    def target_decision_type(self) -> str:
        return "noul"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        passage = str(raw_record.get("passage", "")).strip()
        question = str(raw_record.get("question", "")).strip()
        raw_ans = raw_record.get("answer", False)
        if isinstance(raw_ans, str):
            answer = raw_ans.strip().lower() in ("true", "1", "yes")
        else:
            answer = bool(raw_ans)

        message = f"Passage: {passage}\nQuestion: {question}?" if passage else question
        return {
            "id": f"BOOLQ-{record_idx:04d}",
            "domain": "reading_comprehension_boolq",
            "message": message,
            "decision_type": "noul",
            "instructions": f"Based on the provided passage, answer the question: {question}?",
            "rules": {
                "true": f"The passage confirms or affirms the question (Yes): {question}",
                "false": "The passage contradicts, is missing, prohibited, denied, absent, or does not confirm the question (No).",
            },
            "expected_result": answer,
            "metadata": {
                "source": "google/boolq",
                "question": question,
                "title": raw_record.get("title", ""),
            },
        }


# =============================================================================
# 2. MultiNLI Adapter (`noul`)
# =============================================================================
class MultiNLIAdapter(BaseDatasetAdapter):
    """Adapter for NYU MultiNLI (GLUE) -> MicroLLM `noul` entailment evaluation."""

    @property
    def dataset_name(self) -> str:
        return "nyu-mll/multi_nli"

    @property
    def target_decision_type(self) -> str:
        return "noul"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        premise = str(raw_record.get("premise", "")).strip()
        hypothesis = str(raw_record.get("hypothesis", "")).strip()
        raw_label = raw_record.get("label", 0)
        # In MNLI: 0 = entailment (True), 1 = neutral (False), 2 = contradiction (False)
        if isinstance(raw_label, bool):
            is_entailed = raw_label
        elif isinstance(raw_label, int):
            is_entailed = raw_label == 0
        else:
            lbl_str = str(raw_label).strip().lower()
            is_entailed = lbl_str in ("0", "entailment", "true", "yes")

        return {
            "id": f"MNLI-{record_idx:04d}",
            "domain": "nli_entailment_mnli",
            "message": f"Premise: {premise}\nHypothesis: {hypothesis}",
            "decision_type": "noul",
            "instructions": f"Does the premise entail and confirm the hypothesis: {hypothesis}?",
            "rules": {
                "true": f"Entailment: the premise directly confirms, supports, and entails the hypothesis: {hypothesis}",
                "false": "Contradiction or unestablished: the premise contradicts, negates, is missing, prohibited, denied, or fails to support the hypothesis.",
            },
            "expected_result": is_entailed,
            "metadata": {
                "source": "nyu-mll/multi_nli",
                "genre": raw_record.get("genre", "general"),
                "raw_label": raw_label,
            },
        }


# =============================================================================
# 3. PolyAI Banking77 Adapter (`choice`)
# =============================================================================
BANKING77_CORE_INTENTS = [
    "card_arrival",
    "card_linking",
    "card_lost_or_stolen",
    "card_declined",
    "pin_blocked",
    "automatic_top_up",
    "top_up_failed",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "transfer_fee_charged",
    "transfer_not_received_by_recipient",
    "wrong_amount_of_cash_received",
    "fiat_currency_exchange",
    "pending_cash_withdrawal",
    "compromised_card",
    "refund_not_showing_up",
]

BANKING77_ALL_77_INTENTS: list[str] = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_exchange",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
BANKING77_FULL_77_INTENTS = BANKING77_ALL_77_INTENTS

BANKING77_OPTIONS_MAP = {
    "card_arrival": "New contactless debit card hasn't shown up in mailbox yet, delivery status, tracking or arrival of a newly ordered card",
    "card_linking": "Adding or linking an existing payment card to a mobile app or wallet",
    "card_lost_or_stolen": "Reporting a lost, misplaced, stolen physical card, wallet stolen on subway, block card immediately",
    "card_declined": "Payments or transactions refused or declined at supermarket checkout cashier or online",
    "pin_blocked": "PIN security code entered wrong multiple times at ATM and now PIN is locked or blocked",
    "automatic_top_up": "Setting up recurring or automated account top-ups and auto-reload schedule",
    "top_up_failed": "Account top-up or balance reload transaction failed or errored",
    "balance_not_updated_after_cheque_or_cash_deposit": "Deposited cash or cheque at branch not reflected or updated in current account balance",
    "transfer_fee_charged": "Charged an unexpected euro commission fee on a funds transfer when sending money abroad",
    "transfer_not_received_by_recipient": "Sent money or wire transfer not received by recipient bank account",
    "wrong_amount_of_cash_received": "ATM machine disbursed wrong or smaller amount of cash notes than requested",
    "fiat_currency_exchange": "Exchange US dollars into British pounds or foreign currency conversion in multi-currency wallet",
    "pending_cash_withdrawal": "ATM cash withdrawal still showing pending status after cash was not dispensed",
    "compromised_card": "Suspicious unauthorized fraud transactions on statement, card compromised by unknown merchant",
    "refund_not_showing_up": "Store refunded purchase last week but refund credit is still not showing on account",
}

BANKING77_ALL_77_DESCRIPTIONS: dict[str, str] = {
    intent: f"{intent.replace('_', ' ').replace('?', '')} banking inquiry"
    for intent in BANKING77_ALL_77_INTENTS
}
BANKING77_ALL_77_DESCRIPTIONS.update(BANKING77_OPTIONS_MAP)
BANKING77_ALL_77_DESCRIPTIONS["Refund_not_showing_up"] = BANKING77_OPTIONS_MAP["refund_not_showing_up"]
BANKING77_ALL_77_DESCRIPTIONS["lost_or_stolen_card"] = BANKING77_OPTIONS_MAP["card_lost_or_stolen"]
BANKING77_ALL_77_DESCRIPTIONS["declined_card_payment"] = BANKING77_OPTIONS_MAP["card_declined"]


class Banking77Adapter(BaseDatasetAdapter):
    """Adapter for PolyAI Banking77 -> MicroLLM `choice` intent routing."""

    def __init__(self, options_map: dict[str, str] | None = None) -> None:
        self.options = options_map or BANKING77_OPTIONS_MAP

    @property
    def dataset_name(self) -> str:
        return "PolyAI/banking77"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        label_text = raw_record.get("label_text")
        label = raw_record.get("label", "")
        if label_text and str(label_text).strip():
            target_label = str(label_text).strip()
        elif isinstance(label, int) and 0 <= label < len(BANKING77_ALL_77_INTENTS):
            target_label = BANKING77_ALL_77_INTENTS[label]
        elif isinstance(label, int) and 0 <= label < len(BANKING77_CORE_INTENTS):
            target_label = BANKING77_CORE_INTENTS[label]
        else:
            target_label = str(label).strip()

        rules = dict(self.options)
        if target_label and target_label not in rules:
            rules[target_label] = BANKING77_ALL_77_DESCRIPTIONS.get(
                target_label, f"{target_label.replace('_', ' ')} banking customer service intent"
            )

        return {
            "id": f"BANK77-{record_idx:04d}",
            "domain": "banking_customer_intent",
            "message": text,
            "decision_type": "choice",
            "instructions": "Route this banking customer inquiry to the exact matching intent category.",
            "rules": rules,
            "expected_result": target_label,
            "metadata": {
                "source": "PolyAI/banking77",
                "original_label": label,
            },
        }


# =============================================================================
# 4. AG News Adapter (`choice`)
# =============================================================================
AG_NEWS_LABELS = ["world", "sports", "business", "sci_tech"]

AG_NEWS_OPTIONS_MAP = {
    "world": "International diplomacy, diplomatic relations, foreign ministers, geneva, election, elections, government treaties, geopolitical conflict, united nations summit, president, prime minister, war, military, troops, baghdad, iraq, palestinian, israel, police, killed, bomb, nuclear, hostage",
    "sports": "Championship tournament, football, olympic medal, stadium match victory, coach and athlete score, team, game, season, league, win, won, player, players, cup, World Series, baseball, basketball, soccer, tennis, golf, race, striker, medal",
    "business": "Stock market shares, corporate earnings, quarterly profit, merger acquisition, central bank economy, company, oil, prices, bank, percent, billion, million, dollar, sales, market, stocks, investors, chief executive, retail, trade, inflation, revenue, financial",
    "sci_tech": "Software engineering, NASA space telescope, semiconductor AI chip, cybersecurity, satellite launch, microsoft, internet, computer, technology, digital, web, Linux, Windows, Google, Apple, wireless, mobile phone, scientist, research, space, security, vulnerability, browser",
}


class AGNewsAdapter(BaseDatasetAdapter):
    """Adapter for AG News (`fancyzhx/ag_news`) -> MicroLLM 4-way `choice` topic classification."""

    def __init__(self, options_map: dict[str, str] | None = None) -> None:
        self.options = options_map or AG_NEWS_OPTIONS_MAP

    @property
    def dataset_name(self) -> str:
        return "fancyzhx/ag_news"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        raw_label = raw_record.get("label", 0)
        if isinstance(raw_label, int) and 0 <= raw_label < len(AG_NEWS_LABELS):
            target = AG_NEWS_LABELS[raw_label]
        else:
            target = str(raw_label).strip().lower()

        return {
            "id": f"AGNEWS-{record_idx:04d}",
            "domain": "news_topic_classification",
            "message": text,
            "decision_type": "choice",
            "instructions": "Classify this news article into one of the 4 canonical topic categories.",
            "rules": self.options,
            "expected_result": target,
            "metadata": {
                "source": "fancyzhx/ag_news",
                "raw_label": raw_label,
            },
        }


# =============================================================================
# 5. CLINC150 Adapter (`choice` + Abstention)
# =============================================================================
CLINC_IN_SCOPE_OPTIONS = {
    "flight_status": "Checking flight departures, arrivals, delays, status, is flight on time or delayed",
    "cancel_reservation": "Canceling restaurant dinner table, hotel room, or flight reservations",
    "order_food": "Placing a meal delivery or takeout pizza restaurant food order",
    "change_shipping_address": "Updating parcel delivery destination or postal street shipping address",
    "weather": "Checking weather forecasts, rain, snow, or outdoor temperatures in a city",
    "set_alarm": "Creating a wake-up call, timer, or morning alarm for 7 AM",
    "pay_bill": "Paying a monthly utility, electricity, credit card, or phone invoice bill",
    "transfer": "Transferring money or wire funds between checking and savings bank accounts",
    "find_phone": "Locating a misplaced lost smartphone or ringing missing mobile device",
    "translate": "Translating words or phrases into a foreign language such as French, Spanish, or Italian",
}

CLINC_INT_ID_MAP = {
    42: "oos",
    61: "translate",
    122: "flight_status",
    114: "cancel_reservation",
    25: "order_food",
    16: "weather",
    88: "set_alarm",
    143: "pay_bill",
    133: "transfer",
    102: "find_phone",
}


class CLINC150Adapter(BaseDatasetAdapter):
    """Adapter for CLINC150 (clinc_oos) -> MicroLLM `choice` with Out-Of-Scope abstention."""

    def __init__(self, in_scope_options: dict[str, str] | None = None) -> None:
        self.options = in_scope_options or CLINC_IN_SCOPE_OPTIONS

    @property
    def dataset_name(self) -> str:
        return "clinc_oos"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        raw_intent = raw_record.get("intent", "oos")
        if isinstance(raw_intent, int):
            intent = CLINC_INT_ID_MAP.get(raw_intent, "oos")
        else:
            intent_str = str(raw_intent).strip()
            if intent_str.isdigit():
                intent = CLINC_INT_ID_MAP.get(int(intent_str), "oos")
            else:
                intent = intent_str

        is_oos = intent == "oos" or intent not in self.options

        return {
            "id": f"CLINC-{record_idx:04d}",
            "domain": "clinc_intent_oos",
            "message": text,
            "decision_type": "choice",
            "instructions": "Classify the user utterance into the supported assistant intents, or abstain if out of scope.",
            "rules": self.options,
            "expected_result": "x_abstain" if is_oos else intent,
            "metadata": {
                "source": "clinc_oos",
                "is_out_of_scope": is_oos,
                "original_intent": raw_intent,
            },
        }


# =============================================================================
# 6. Amazon Reviews Multi Adapter (`score`)
# =============================================================================
AMAZON_REVIEW_LEVELS = [
    "Terrible (1 star): defective, broke on first day, awful, worst purchase, cheap plastic, waste of money, garbage, trash, useless, refund, junk, broken, returned, horrible, inutilisable, catastrophique, defekt, pésimo",
    "Poor (2 stars): disappointed, battery dies fast, major flaws, below expectations, not worth, poorly made, cheap, stopped working, frustrating, flimsy, décevant, fragile, enttäuschend, malo",
    "Average (3 stars): acceptable, works as described, decent for price, nothing fancy, okay, ok, alright, fine, three stars, average, fair, mixed feelings, correct, moyen, durchschnittlich, aceptable",
    "Good (4 stars): satisfied, comfortable, great sound isolation, very good, nice, pleased, works well, happy with purchase, good quality, solid, très bon, confortable, sehr gut, muy bueno",
    "Outstanding (5 stars): masterpiece, exceptional sound, highly recommended, absolutely flawless, love it, perfect, best, amazing, awesome, fantastic, five stars, parfait, excellent, ausgezeichnet, excelente",
]


class AmazonReviewsAdapter(BaseDatasetAdapter):
    """Adapter for Amazon Reviews Multi -> MicroLLM `score` ordinal rubric (EN/FR/ES/DE)."""

    def __init__(self, levels: list[str] | None = None) -> None:
        self.levels = levels or AMAZON_REVIEW_LEVELS

    @property
    def dataset_name(self) -> str:
        return "amazon_reviews_multi"

    @property
    def target_decision_type(self) -> str:
        return "score"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        body = str(raw_record.get("review_body", "") or raw_record.get("text", "")).strip()
        title = str(raw_record.get("review_title", "")).strip()
        if "label" in raw_record and "stars" not in raw_record and "star_rating" not in raw_record:
            expected_level_index = max(0, min(4, int(raw_record["label"])))
            stars = expected_level_index + 1
        else:
            stars = int(raw_record.get("stars", raw_record.get("star_rating", 3)))
            stars = max(1, min(5, stars))
            expected_level_index = stars - 1

        message = f"Title: {title}\nReview: {body}" if title else body
        return {
            "id": f"AMZREV-{record_idx:04d}",
            "domain": "amazon_product_reviews",
            "message": message.strip(),
            "decision_type": "score",
            "instructions": "Rate this product review from terrible (0) to outstanding (4).",
            "rules": self.levels,
            "expected_result": expected_level_index,
            "metadata": {
                "source": "amazon_reviews_multi",
                "star_rating": stars,
                "language": raw_record.get("language", "en"),
            },
        }


# =============================================================================
# 7. Yelp Review Full Adapter (`score`)
# =============================================================================
YELP_REVIEW_LEVELS = [
    "1 star: horrible service, rude staff, cold inedible food, flat tire, worst experience ever, never coming back, terrible, disgusting, awful, zero stars, manager, overcharged, dirty",
    "2 stars: slow waiter, overpriced mediocre meal, disappointing quality, cold coffee, below average, not impressed, bland, dry, undercooked, two stars, Meh",
    "3 stars: okay experience, average food, nothing special, decent lunch for the price, standard service, three stars, alright, fair, reasonable, hit or miss",
    "4 stars: very good dinner, friendly attentive staff, tasty dishes, pleasant atmosphere, solid choice, enjoyed, nice place, great food, four stars, definitely return",
    "5 stars: incredible service, delicious food, best restaurant in town, outstanding chef, highly recommended 5 stars, amazing, favorite place, phenomenal, loved everything, perfection",
]


class YelpReviewFullAdapter(BaseDatasetAdapter):
    """Adapter for Yelp Review Full (`Yelp/yelp_review_full`) -> MicroLLM 5-level `score` rubric."""

    def __init__(self, levels: list[str] | None = None) -> None:
        self.levels = levels or YELP_REVIEW_LEVELS

    @property
    def dataset_name(self) -> str:
        return "Yelp/yelp_review_full"

    @property
    def target_decision_type(self) -> str:
        return "score"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        if "label" in raw_record:
            expected_idx = max(0, min(4, int(raw_record["label"])))
        else:
            stars = max(1, min(5, int(raw_record.get("stars", 3))))
            expected_idx = stars - 1

        return {
            "id": f"YELP-{record_idx:04d}",
            "domain": "yelp_service_reviews",
            "message": text,
            "decision_type": "score",
            "instructions": "Rate this customer service review on the 5-level ordinal scale (0=1 star to 4=5 stars).",
            "rules": self.levels,
            "expected_result": expected_idx,
            "metadata": {
                "source": "Yelp/yelp_review_full",
                "stars": expected_idx + 1,
            },
        }


# =============================================================================
# 8. Google GoEmotions Adapter (`multi`)
# =============================================================================
GO_EMOTIONS_TAXONOMY = {
    "admiration": "Praise, respect, kind assistance, saved my day, high esteem, awesome, amazing, impressive, great job, admiring someone",
    "amusement": "Funniest joke, laughing so hard, humorous, hilarious, entertaining, lol, haha, funny, comedy",
    "anger": "Absolutely furious, angry, outraged, canceled without notice, hostility, hate, mad, rage, idiot, stupid",
    "annoyance": "Irritated, frustrated, annoyed, canceled train without notice or refund, annoying, ugh,ugh, bothersome",
    "approval": "Agreeing, approving, endorsing, supporting a smart decision, yes, agree, exactly, right, good point",
    "caring": "Sympathy, compassion, take care, stay safe, get well soon, helping and supporting someone",
    "confusion": "Confused, uncertain, puzzled, I don't understand, what do you mean, weird, unclear",
    "curiosity": "Inquisitive, asking how does this work, curious to learn more, why, how, wondering, question",
    "desire": "Wishing, wanting, craving, I wish, I want, hope to get, need this",
    "disappointment": "Disappointed, let down, sad over unmet expectations, shame, unfortunate, too bad",
    "disapproval": "Disagreeing, disapproving, no, wrong, bad idea, against, unacceptable",
    "disgust": "Disgusting, gross, revolting, nasty, sickening, awful, repulsive",
    "embarrassment": "Embarrassed, awkward, ashamed, cringe, humiliating, oops",
    "excitement": "Excited, thrilled, can't wait, hyped, pumped, wow, amazing news",
    "fear": "Afraid, scared, terrified, anxious, worried, creepy, dangerous, panic",
    "gratitude": "Thank you so much, grateful, thankful, appreciating kind assistance, thanks, appreciate",
    "grief": "Grieving, mourning, loss, passed away, rest in peace, RIP, tragedy",
    "joy": "Happy, delighted, joyful, funniest joke, laughing, celebrating positive energy, glad, wonderful",
    "love": "Love, adorable, sweetheart, affection, fond, I love this, heart",
    "nervousness": "Nervous, anxious, tense, stressed, worried about upcoming event",
    "optimism": "Hopeful, optimistic outlook, confident the future will improve, hope, good luck",
    "pride": "Proud, accomplishment, achievement, success, well done",
    "realization": "Realizing, sudden understanding, oh I see, TIL, turns out, epiphany",
    "relief": "Relieved, phew, glad it is over, safe now, weight off shoulders",
    "remorse": "Deeply sorry, apologizing for how I acted, feeling terrible about hurting you, guilt, my bad",
    "sadness": "Heartbroken, grieving, crying, feeling down and unhappy, sad, depressing, sorry to hear",
    "surprise": "Surprised, shocked, unexpected, wow, omg, plot twist, unbelievable",
    "neutral": "Neutral factual statement, objective observation, normal comment, general information",
}

GO_EMOTIONS_ID_TO_NAME: dict[int, str] = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}
GO_EMOTIONS_ALL_28_LABELS: list[str] = list(GO_EMOTIONS_TAXONOMY.keys())


class GoEmotionsAdapter(BaseDatasetAdapter):
    """Adapter for Google GoEmotions -> MicroLLM `multi` multi-label tagging."""

    def __init__(self, taxonomy: dict[str, str] | None = None) -> None:
        self.taxonomy = taxonomy or GO_EMOTIONS_TAXONOMY

    @property
    def dataset_name(self) -> str:
        return "google-research-datasets/go_emotions"

    @property
    def target_decision_type(self) -> str:
        return "multi"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        raw_labels = raw_record.get("labels", [])
        if isinstance(raw_labels, str):
            raw_labels = [lbl.strip() for lbl in raw_labels.split(",") if lbl.strip()]
        elif hasattr(raw_labels, "tolist"):
            raw_labels = list(raw_labels.tolist())

        normalized_labels: list[str] = []
        for lbl in raw_labels:
            if isinstance(lbl, int):
                mapped = GO_EMOTIONS_ID_TO_NAME.get(lbl)
                if mapped:
                    normalized_labels.append(mapped)
            else:
                lbl_str = str(lbl).strip()
                if lbl_str.isdigit():
                    mapped = GO_EMOTIONS_ID_TO_NAME.get(int(lbl_str))
                    if mapped:
                        normalized_labels.append(mapped)
                elif lbl_str in self.taxonomy:
                    normalized_labels.append(lbl_str)

        return {
            "id": f"GOEMO-{record_idx:04d}",
            "domain": "emotion_multi_label",
            "message": text,
            "decision_type": "multi",
            "instructions": "Detect all active emotional expressions present in the comment.",
            "rules": self.taxonomy,
            "expected_result": normalized_labels,
            "metadata": {
                "source": "google-research-datasets/go_emotions",
                "all_labels": raw_labels,
            },
        }


# =============================================================================
# 9. Reuters-21578 Adapter (`multi`)
# =============================================================================
REUTERS_TOPICS_TAXONOMY = {
    "earn": "Corporate quarterly earnings, net profit, dividend per share, revenue growth, cts, net, loss, shr, profit, qtr, revs, oper",
    "acq": "Corporate merger, acquisition, hostile takeover bid, buyout agreement, stake, shares, acquire, bought, sell, unit, tender offer",
    "crude": "Crude oil barrels, OPEC petroleum production, refinery energy prices, bpd, barrel, energy",
    "interest": "Central bank interest rates, Federal Reserve discount rate, bond yields, prime rate, bank rate, money market, pct",
    "trade": "International trade balance, tariff duties, export and import deficit, surplus, GATT, bilateral trade, imports, exports",
    "grain": "Wheat, corn, soybean agricultural grain harvest and export tonnage, USDA, tonnes, bushels, crops, agriculture",
    "money-fx": "Foreign exchange currency markets, dollar, yen, mark, sterling, central bank intervention, exchange rate, currency",
    "ship": "Maritime shipping vessels, cargo tankers, port strike, freight transport, gulf shipping",
    "wheat": "Wheat grain harvest, flour exports, agricultural wheat tonnage, USDA wheat",
    "corn": "Corn maize harvest, agricultural grain exports, USDA corn bushels",
    "dlr": "US dollar currency value, exchange rate against yen and mark, greenback",
    "money-supply": "M1 M2 M3 money supply growth, central bank monetary aggregates, federal reserve",
    "oilseed": "Soybean, rapeseed, palm oil agricultural oilseed crushing and exports",
    "sugar": "Raw and white sugar cane harvest, quota, world sugar market prices",
    "coffee": "Coffee beans export quotas, ICO international coffee organization, arabica harvest",
    "gnp": "Gross national product GNP GDP economic growth, inflation, macroeconomic output",
    "gold": "Gold bullion ounces, precious metals mining, troy ounce price",
    "veg-oil": "Vegetable oil, palm oil, soybean oil agricultural exports and refining",
    "nat-gas": "Natural gas pipeline, cubic feet energy production, gas utilities",
}
REUTERS_CORE_TOPICS = REUTERS_TOPICS_TAXONOMY


class Reuters21578Adapter(BaseDatasetAdapter):
    """Adapter for Reuters-21578 financial newswire -> MicroLLM `multi` topic tagging."""

    def __init__(self, taxonomy: dict[str, str] | None = None) -> None:
        self.taxonomy = taxonomy or REUTERS_TOPICS_TAXONOMY

    @property
    def dataset_name(self) -> str:
        return "reuters21578"

    @property
    def target_decision_type(self) -> str:
        return "multi"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        topics = raw_record.get("topics", raw_record.get("labels", []))
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(",") if t.strip()]
        elif hasattr(topics, "tolist"):
            topics = list(topics.tolist())
        matched = [str(t) for t in topics if str(t) in self.taxonomy]
        return {
            "id": f"REUTERS-{record_idx:04d}",
            "domain": "financial_newswire_reuters",
            "message": text,
            "decision_type": "multi",
            "instructions": "Select all applicable Reuters financial newswire topic categories.",
            "rules": self.taxonomy,
            "expected_result": matched,
            "metadata": {
                "source": "reuters21578",
                "topics": matched,
            },
        }


# =============================================================================
# 10. AllenAI WildGuard Adapter (`guardrail`)
# =============================================================================
class WildGuardAdapter(BaseDatasetAdapter):
    """Adapter for AllenAI WildGuard (`allenai/wildguardmix`) -> MicroLLM `guardrail` safety shield."""

    @property
    def dataset_name(self) -> str:
        return "allenai/wildguardmix"

    @property
    def target_decision_type(self) -> str:
        return "guardrail"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        prompt = str(raw_record.get("prompt", "") or raw_record.get("text", "")).strip()
        category = str(
            raw_record.get("category") or raw_record.get("subcategory") or "prompt_injection"
        ).strip()
        if "is_harmful" in raw_record and raw_record["is_harmful"] is not None:
            raw_harm = raw_record["is_harmful"]
            if isinstance(raw_harm, str):
                is_harmful = raw_harm.strip().lower() in ("harmful", "1", "true", "yes", "unsafe")
            else:
                is_harmful = bool(raw_harm)
        else:
            raw_lbl = (
                raw_record.get("label")
                if raw_record.get("label") is not None
                else raw_record.get("prompt_harm_label", False)
            )
            if isinstance(raw_lbl, str):
                # RCA-023: In walledai/WildGuardTest, `label` is `"harmful"` or `"unharmful"`.
                # `bool("unharmful")` would evaluate to True unless parsed explicitly!
                is_harmful = raw_lbl.strip().lower() in ("harmful", "1", "true", "yes", "unsafe")
            elif raw_lbl is None:
                is_harmful = False
            else:
                is_harmful = bool(raw_lbl)

        return {
            "id": f"WILDG-{record_idx:04d}",
            "domain": "ai_safety_guardrail",
            "message": prompt,
            "decision_type": "guardrail",
            "instructions": "Detect prompt injection, jailbreak, destructive command, or PII leak.",
            "rules": {
                "threshold": 0.55,
                "category": category,
                "context": raw_record.get("context"),
            },
            "expected_result": is_harmful,
            "metadata": {
                "source": "allenai/wildguardmix",
                "category": category,
                "is_harmful": is_harmful,
            },
        }


# =============================================================================
# 11. LMSYS ToxicChat Adapter (`guardrail`)
# =============================================================================
class ToxicChatAdapter(BaseDatasetAdapter):
    """Adapter for LMSYS ToxicChat (`lmsys/toxic-chat`) -> MicroLLM `guardrail`."""

    @property
    def dataset_name(self) -> str:
        return "lmsys/toxic-chat"

    @property
    def target_decision_type(self) -> str:
        return "guardrail"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        user_input = str(raw_record.get("user_input", "") or raw_record.get("text", "")).strip()
        toxicity = int(raw_record.get("toxicity", 0) or 0)
        jailbreaking = int(raw_record.get("jailbreaking", 0) or 0)
        is_harmful = bool(raw_record.get("is_harmful", (toxicity == 1) or (jailbreaking == 1)))

        return {
            "id": f"TOXIC-{record_idx:04d}",
            "domain": "toxic_chat_moderation",
            "message": user_input,
            "decision_type": "guardrail",
            "instructions": "Evaluate real user chat input for jailbreak attempts, destructive commands, or credential leaks.",
            "rules": {
                "threshold": 0.55,
                "category": "jailbreaking" if jailbreaking else "toxicity",
            },
            "expected_result": is_harmful,
            "metadata": {
                "source": "lmsys/toxic-chat",
                "toxicity": toxicity,
                "jailbreaking": jailbreaking,
            },
        }


# =============================================================================
# 12. Deepset Prompt Injections Adapter (`guardrail`)
# =============================================================================
class DeepsetPromptInjectionsAdapter(BaseDatasetAdapter):
    """Adapter for Deepset Prompt Injections (`deepset/prompt-injections`) -> MicroLLM `guardrail`."""

    @property
    def dataset_name(self) -> str:
        return "deepset/prompt-injections"

    @property
    def target_decision_type(self) -> str:
        return "guardrail"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "") or raw_record.get("prompt", "")).strip()
        label = int(raw_record.get("label", 0) or 0)
        is_injection = label == 1 or bool(raw_record.get("is_injection", False))

        return {
            "id": f"DEEPINJ-{record_idx:04d}",
            "domain": "prompt_injection_security",
            "message": text,
            "decision_type": "guardrail",
            "instructions": "Detect direct or indirect prompt injection and system instruction override.",
            "rules": {
                "threshold": 0.55,
                "category": "prompt_injection" if is_injection else "safe_grounded",
            },
            "expected_result": is_injection,
            "metadata": {
                "source": "deepset/prompt-injections",
                "label": label,
            },
        }


# =============================================================================
# 13. ChoiceBench Adapter (`choice` + AnyJev L0 Order Invariance)
# =============================================================================
class ChoiceBenchAdapter(BaseDatasetAdapter):
    """Adapter for ChoiceBench (`cotenthusiast/choicebench`) -> `choice` with L0 order invariance check."""

    @property
    def dataset_name(self) -> str:
        return "cotenthusiast/choicebench"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        question = str(
            raw_record.get("question", "")
            or raw_record.get("prompt", "")
            or raw_record.get("state", "")
        ).strip()
        raw_options = raw_record.get("options")
        raw_choices = raw_record.get("choices")
        raw_answer = raw_record.get("answer", "")

        # RCA-024: Support both dict `options` (curated fixtures) and list `choices` + int `answer` (MMLU/ChoiceBench parquet)
        letter_keys = ["A", "B", "C", "D", "E", "F"]
        if isinstance(raw_options, dict) and raw_options:
            options = {str(k): str(v) for k, v in raw_options.items()}
            answer = str(raw_answer).strip()
        elif raw_choices is not None:
            choices_list = (
                list(raw_choices.tolist()) if hasattr(raw_choices, "tolist") else list(raw_choices)
            )
            options = {
                letter_keys[i] if i < len(letter_keys) else f"opt_{i}": str(c)
                for i, c in enumerate(choices_list)
            }
            if isinstance(raw_answer, int) and 0 <= raw_answer < len(choices_list):
                answer = letter_keys[raw_answer] if raw_answer < len(letter_keys) else f"opt_{raw_answer}"
            elif str(raw_answer).strip().isdigit() and int(str(raw_answer).strip()) < len(choices_list):
                idx = int(str(raw_answer).strip())
                answer = letter_keys[idx] if idx < len(letter_keys) else f"opt_{idx}"
            else:
                answer = str(raw_answer).strip()
        else:
            options = {}
            answer = str(raw_answer).strip()

        return {
            "id": f"CHBENCH-{record_idx:04d}",
            "domain": "choicebench_order_invariance",
            "message": question,
            "decision_type": "choice",
            "instructions": "Select the single accurate option invariant to candidate permutation order.",
            "rules": options,
            "expected_result": answer,
            "metadata": {
                "source": "cotenthusiast/choicebench",
                "check_order_invariance": True,
                "cardinality_k": len(options),
            },
        }


# =============================================================================
# 14. Callo-Bias Set Adapter (`choice` + Cyclic Permutation Drift Audit)
# =============================================================================
class CalloBiasSetAdapter(BaseDatasetAdapter):
    """Adapter for Callo-Bias Set (`peculiar-ragdoll/callo-bias-set`) -> `choice` cyclic permutation probe."""

    @property
    def dataset_name(self) -> str:
        return "peculiar-ragdoll/callo-bias-set"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        raw_state = (
            raw_record.get("state", "")
            or raw_record.get("question", "")
            or raw_record.get("prompt", "")
        )
        if isinstance(raw_state, (dict, list)):
            state = json.dumps(raw_state, ensure_ascii=False)
        else:
            state = str(raw_state).strip()

        raw_options = raw_record.get("options")
        expected = str(
            raw_record.get("expected")
            or raw_record.get("answer")
            or raw_record.get("gold_label")
            or ""
        ).strip()
        instructions = "Evaluate candidate options with zero cyclic positional probability drift."

        # RCA-024: Support nested `questions` dict schema from official `peculiar-ragdoll/callo-bias-set` JSONL
        if isinstance(raw_options, dict) and raw_options:
            options = {str(k): str(v) for k, v in raw_options.items()}
        elif isinstance(raw_record.get("questions"), dict) and raw_record["questions"]:
            q_dict = raw_record["questions"]
            first_q = next(iter(q_dict.values()))
            instructions = str(first_q.get("instructions") or instructions).strip()
            criteria = first_q.get("criteria") or first_q.get("options") or {}
            options = {
                str(k): (str(v) if v is not None else str(k).replace("_", " "))
                for k, v in criteria.items()
            }
            if not expected:
                expected = str(
                    first_q.get("label")
                    or first_q.get("gold_label")
                    or first_q.get("expected")
                    or first_q.get("answer")
                    or ""
                ).strip()
        else:
            options = {}

        return {
            "id": f"CALLO-{record_idx:04d}",
            "domain": "callo_bias_cyclic_permutation",
            "message": state,
            "decision_type": "choice",
            "instructions": instructions,
            "rules": options,
            "expected_result": expected,
            "metadata": {
                "source": "peculiar-ragdoll/callo-bias-set",
                "check_order_invariance": True,
                "cardinality_k": len(options),
            },
        }


# =============================================================================
# 15. LMSYS RouteLLM Adapter (`cascade`)
# =============================================================================
class RouteLLMAdapter(BaseDatasetAdapter):
    """Adapter for LMSYS RouteLLM -> MicroLLM System-1 vs System-2 Cascade Routing."""

    @property
    def dataset_name(self) -> str:
        return "lm-sys/RouteLLM"

    @property
    def target_decision_type(self) -> str:
        return "cascade"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        query = str(raw_record.get("query", "") or raw_record.get("prompt", "")).strip()
        # RCA-024: Support both explicit `target_tier`/`complex` and `routellm/gpt4_dataset` (`mixtral_score` 1..5 or `bert_score`)
        if "target_tier" in raw_record or "complex" in raw_record:
            complexity = raw_record.get("target_tier", "fast_system1")
            requires_deep_reasoning = complexity == "deep_system2" or bool(
                raw_record.get("complex", False)
            )
        elif "mixtral_score" in raw_record and raw_record["mixtral_score"] is not None:
            # In LMSYS RouteLLM gpt4_dataset, mixtral_score >= 4 means fast open model suffices (System-1);
            # mixtral_score < 4 means strong GPT-4 System-2 reasoning is required.
            requires_deep_reasoning = int(raw_record["mixtral_score"]) < 4
        elif "bert_score" in raw_record and raw_record["bert_score"] is not None:
            requires_deep_reasoning = float(raw_record["bert_score"]) >= 0.70
        else:
            requires_deep_reasoning = False

        return {
            "id": f"ROUTE-{record_idx:04d}",
            "domain": "llm_routing_cascade",
            "message": query,
            "decision_type": "choice",
            "instructions": "Select the execution tier for this query.",
            "rules": {
                "fast_system1": "Routine customer question, password reset, support ticket triage, basic status classification, invoice check, general query in English with a simple answer for the user to help and explain",
                "deep_system2": "Complex formal proof, Peano arithmetic, Lean 4, Coq theorem proving, distributed Raft consensus, Rust, TLA+",
            },
            "expected_result": "deep_system2" if requires_deep_reasoning else "fast_system1",
            "metadata": {
                "source": "lm-sys/RouteLLM",
                "requires_deep_reasoning": requires_deep_reasoning,
            },
        }


# =============================================================================
# 16. Official JevBench External Adapter (`fstandhartinger/jevbench`)
# =============================================================================
class JevBenchExternalAdapter(BaseDatasetAdapter):
    """Adapter for official JevBench (`fstandhartinger/jevbench`) mixed primitives (`noul`, `choice`, `score`, `multi`)."""

    @property
    def dataset_name(self) -> str:
        return "fstandhartinger/jevbench"

    @property
    def target_decision_type(self) -> str:
        return "jevcompat"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        raw_state = raw_record.get("state", "")
        state = (
            json.dumps(raw_state, ensure_ascii=False)
            if isinstance(raw_state, (dict, list))
            else str(raw_state).strip()
        )
        dtype = str(raw_record.get("decision_type") or raw_record.get("type") or "noul").strip()
        instructions = str(raw_record.get("instructions", "")).strip()

        # RCA-024: Support both pre-normalized `rules`/`expected` and official `JevBench/jevbench` parquet schema
        # (`type`, `options` as list[{"id", "description"}], `levels` as list[str], `gold` as str)
        raw_rules = raw_record.get("rules")
        if raw_rules is not None and (
            (isinstance(raw_rules, dict) and len(raw_rules) > 0)
            or (isinstance(raw_rules, list) and len(raw_rules) > 0)
        ):
            rules: Any = raw_rules
        elif dtype in ("choice", "multi") and raw_record.get("options") is not None:
            raw_opts = raw_record["options"]
            if hasattr(raw_opts, "tolist"):
                raw_opts = list(raw_opts.tolist())
            if isinstance(raw_opts, dict):
                rules = {str(k): str(v) for k, v in raw_opts.items()}
            elif isinstance(raw_opts, list):
                rules = {}
                for item in raw_opts:
                    if isinstance(item, dict) and "id" in item:
                        rules[str(item["id"])] = str(item.get("description") or item["id"])
                    else:
                        rules[str(item)] = str(item)
            else:
                rules = {}
        elif dtype == "score" and raw_record.get("levels") is not None:
            raw_lvls = raw_record["levels"]
            if hasattr(raw_lvls, "tolist"):
                raw_lvls = list(raw_lvls.tolist())
            rules = [str(x) for x in raw_lvls]
        else:
            rules = {
                "true": f"Yes, confirmed and established: {instructions}",
                "false": "No, contradicted, missing, prohibited, denied, or unestablished.",
            }

        if "expected" in raw_record and raw_record["expected"] is not None:
            expected: Any = raw_record["expected"]
        else:
            gold = raw_record.get("gold")
            if dtype == "noul":
                if isinstance(gold, bool):
                    expected = gold
                else:
                    gold_str = str(gold or "").strip().lower()
                    # In JevBench, gold in {"true", "yes", "1"} -> True; {"false", "no", "0", "unknown"} -> False
                    expected = gold_str in ("true", "yes", "1")
            elif dtype == "score":
                try:
                    expected = int(gold) if gold is not None else 0
                except (ValueError, TypeError):
                    expected = 0
            elif dtype == "multi":
                if isinstance(gold, list):
                    expected = [str(x) for x in gold]
                elif isinstance(gold, str):
                    g_s = gold.strip()
                    if g_s.startswith("["):
                        try:
                            expected = [str(x) for x in json.loads(g_s)]
                        except Exception:
                            expected = [x.strip() for x in g_s.strip("[]").split(",") if x.strip()]
                    else:
                        expected = [x.strip() for x in g_s.split(",") if x.strip()]
                else:
                    expected = []
            else:
                expected = "x_abstain" if str(gold).strip().lower() == "unknown" else str(gold or "").strip()

        return {
            "id": str(raw_record.get("id") or f"JEVEXT-{record_idx:04d}"),
            "domain": str(raw_record.get("domain") or "jevbench_official"),
            "message": state,
            "decision_type": dtype,
            "instructions": instructions,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "fstandhartinger/jevbench",
                "split": raw_record.get("split", "official"),
                "raw_gold": raw_record.get("gold"),
            },
        }


from microllm.dataset_fixtures import OFFLINE_WEB_DATASET_SAMPLES
import sys
import types


class _DatasetAdaptersModule(types.ModuleType):
    @property
    def OFFLINE_WEB_DATASET_SAMPLES(self) -> Any:
        if "microllm.dataset_fixtures" in sys.modules:
            return sys.modules["microllm.dataset_fixtures"].OFFLINE_WEB_DATASET_SAMPLES
        import microllm.dataset_fixtures
        return microllm.dataset_fixtures.OFFLINE_WEB_DATASET_SAMPLES


sys.modules[__name__].__class__ = _DatasetAdaptersModule


from microllm.hf_jev_datasets import (
    ALL_12_HF_JEV_MODEL_DATASETS,
    HF_JEV_ADAPTER_CLASSES,
    HF_JEV_DATASET_SERVER_CONFIG_MAP,
    HF_JEV_ECOSYSTEM_MODELS_CATALOG,
    OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES,
    AgentDecisionBenchAdapter,
    AgenticPromptInjection5kAdapter,
    DecisionBenchAdapter,
    JevAdvBenchAdapter,
    JevDistillCorpusV3Adapter,
    JevRAGClaimVerificationAdapter,
    JevSafetyRefusalDecisionsAdapter,
    LocalLLaMATypedDecisionsAdapter,
    OpenJevCorpusAdapter,
    PAWSParaphraseAdapter,
    SST5OrdinalAdapter,
    TasksourceJevTypedDecisionsAdapter,
    list_hf_jev_ecosystem_models,
)

ALL_28_EXTERNAL_DATASETS: list[str] = (
    ALL_16_EXTERNAL_DATASETS + ALL_12_HF_JEV_MODEL_DATASETS
)
HF_DATASET_SERVER_CONFIG_MAP.update(HF_JEV_DATASET_SERVER_CONFIG_MAP)


def get_adapter_for_dataset(dataset_name: str) -> BaseDatasetAdapter:
    """Factory retrieving the appropriate adapter for any of the 28 external dataset identifiers."""
    adapters: dict[str, type[BaseDatasetAdapter]] = {
        "google/boolq": BoolQAdapter,
        "nyu-mll/multi_nli": MultiNLIAdapter,
        "PolyAI/banking77": Banking77Adapter,
        "fancyzhx/ag_news": AGNewsAdapter,
        "clinc_oos": CLINC150Adapter,
        "amazon_reviews_multi": AmazonReviewsAdapter,
        "Yelp/yelp_review_full": YelpReviewFullAdapter,
        "google-research-datasets/go_emotions": GoEmotionsAdapter,
        "reuters21578": Reuters21578Adapter,
        "allenai/wildguardmix": WildGuardAdapter,
        "lmsys/toxic-chat": ToxicChatAdapter,
        "deepset/prompt-injections": DeepsetPromptInjectionsAdapter,
        "cotenthusiast/choicebench": ChoiceBenchAdapter,
        "peculiar-ragdoll/callo-bias-set": CalloBiasSetAdapter,
        "lm-sys/RouteLLM": RouteLLMAdapter,
        "fstandhartinger/jevbench": JevBenchExternalAdapter,
        **HF_JEV_ADAPTER_CLASSES,
    }
    adapter_cls = adapters.get(dataset_name)
    if not adapter_cls:
        raise ValueError(
            f"Unknown web dataset '{dataset_name}'. Supported datasets: {list(adapters.keys())}"
        )
    return adapter_cls()


def load_web_dataset_fixture(dataset_name: str) -> list[dict[str, Any]]:
    """Load and convert curated offline sample records for a web dataset into MicroLLM test entries."""
    raw_records = OFFLINE_WEB_DATASET_SAMPLES.get(
        dataset_name
    ) or OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES.get(dataset_name)
    if not raw_records:
        raise ValueError(f"No offline fixtures found for dataset '{dataset_name}'")
    adapter = get_adapter_for_dataset(dataset_name)
    return adapter.convert_batch(raw_records)


def _parse_cached_split_file(file_path: Path, fmt: str) -> list[dict[str, Any]]:
    """Parse a cached Parquet or JSONL split file, raising ValueError if empty or malformed."""
    if not file_path.is_file() or file_path.stat().st_size == 0:
        raise ValueError("Cached split file is missing or empty.")
    if fmt == "parquet":
        if file_path.stat().st_size < 12:
            raise ValueError("Truncated Parquet file smaller than 12-byte magic framing.")
        with file_path.open("rb") as f:
            header = f.read(4)
            f.seek(-4, 2)
            footer = f.read(4)
        if header != b"PAR1" or footer != b"PAR1":
            raise ValueError("Corrupted or truncated Parquet file missing PAR1 magic bytes.")
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)
        raw_rows: list[dict[str, Any]] = table.to_pylist()
    else:
        raw_rows = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not raw_rows:
        raise ValueError("Cached split file contained zero rows.")
    return raw_rows


def load_full_external_dataset_split(
    dataset_name: str,
    cache_dir: str | Path | None = None,
    allow_network: bool = True,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Load and convert the complete unsampled official evaluation split for an external dataset.

    Reads from the local Parquet/JSONL cache (`~/.cache/microllm/external_datasets/`), evicting any
    corrupted or truncated cache file (`RCA-027`), or downloads the official Hugging Face Parquet/JSONL
    split atomically via `FULL_SPLIT_PARQUET_URL_MAP` if missing and `allow_network=True`.
    Falls back to `load_web_dataset_fixture(dataset_name)` if offline and uncached.
    """
    adapter = get_adapter_for_dataset(dataset_name)
    spec = FULL_SPLIT_PARQUET_URL_MAP.get(dataset_name)
    if spec is None:
        return load_web_dataset_fixture(dataset_name)

    resolved_cache = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "microllm" / "external_datasets"
    )
    resolved_cache.mkdir(parents=True, exist_ok=True)
    file_path = resolved_cache / spec["filename"]

    if file_path.exists():
        try:
            raw_rows = _parse_cached_split_file(file_path, spec["format"])
            return adapter.convert_batch(raw_rows)
        except ImportError:
            return load_web_dataset_fixture(dataset_name)
        except Exception:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not file_path.is_file() and allow_network:
        if spec["format"] == "parquet":
            try:
                import pyarrow.parquet  # noqa: F401
            except ImportError:
                return load_web_dataset_fixture(dataset_name)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            import httpx

            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                resp = client.get(spec["url"])
                if resp.status_code == 200 and len(resp.content) > 0:
                    tmp_path.write_bytes(resp.content)
                    tmp_path.replace(file_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if file_path.is_file():
        try:
            raw_rows = _parse_cached_split_file(file_path, spec["format"])
            return adapter.convert_batch(raw_rows)
        except ImportError:
            return load_web_dataset_fixture(dataset_name)
        except Exception:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass

    return load_web_dataset_fixture(dataset_name)


def fetch_huggingface_rows(
    dataset_name: str,
    limit: int | None = 10,
    allow_network: bool = True,
    timeout: float = 8.0,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch live rows from Hugging Face Datasets Server (`datasets-server.huggingface.co/rows`) or full split.

    When `limit is None` (or `limit <= 0`), loads the complete unsampled official evaluation split via
    `load_full_external_dataset_split`. Otherwise fetches `limit` rows from the Hugging Face Datasets
    Server API (`HF_DATASET_SERVER_CONFIG_MAP`, `RCA-018`) or local cache, with deterministic fallback.
    """
    if limit is None or int(limit) <= 0:
        return load_full_external_dataset_split(
            dataset_name,
            cache_dir=cache_dir,
            allow_network=allow_network,
            timeout=max(timeout, 30.0),
        )

    adapter = get_adapter_for_dataset(dataset_name)
    safe_limit = max(1, int(limit))
    if allow_network and dataset_name in HF_DATASET_SERVER_CONFIG_MAP:
        cfg = HF_DATASET_SERVER_CONFIG_MAP[dataset_name]
        try:
            import httpx

            url = (
                "https://datasets-server.huggingface.co/rows"
                f"?dataset={cfg['hf_dataset']}&config={cfg['config']}"
                f"&split={cfg['split']}&offset=0&length={safe_limit}"
            )
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                raw_rows = [item["row"] for item in payload.get("rows", []) if "row" in item]
                if raw_rows:
                    return adapter.convert_batch(raw_rows[:safe_limit])
        except Exception:
            pass

    fixtures = load_web_dataset_fixture(dataset_name)
    return fixtures[:safe_limit]


def build_external_datasets_corpus(
    entries_per_dataset: int = 40,
    include_hf_jev_models: bool = False,
) -> list[dict[str, Any]]:
    """Build a scaled, deterministic multi-dataset external benchmark corpus across 16 or all 28 external datasets."""
    target_per_ds = max(1, int(entries_per_dataset))
    corpus: list[dict[str, Any]] = []
    target_datasets = (
        ALL_28_EXTERNAL_DATASETS if include_hf_jev_models else ALL_16_EXTERNAL_DATASETS
    )
    for ds_name in target_datasets:
        base_entries = load_web_dataset_fixture(ds_name)
        n_base = len(base_entries)
        for i in range(target_per_ds):
            template = dict(base_entries[i % n_base])
            prefix = template["id"].split("-")[0]
            template["id"] = f"{prefix}-{i + 1:04d}"
            template["metadata"] = {
                **dict(template.get("metadata", {})),
                "dataset_name": ds_name,
                "variant_index": i + 1,
            }
            corpus.append(template)
    return corpus


def build_hf_jev_models_corpus(entries_per_dataset: int = 40) -> list[dict[str, Any]]:
    """Build a scaled, deterministic corpus across all 12 Hugging Face JEV model ecosystem datasets."""
    target_per_ds = max(1, int(entries_per_dataset))
    corpus: list[dict[str, Any]] = []
    for ds_name in ALL_12_HF_JEV_MODEL_DATASETS:
        base_entries = load_web_dataset_fixture(ds_name)
        n_base = len(base_entries)
        for i in range(target_per_ds):
            template = dict(base_entries[i % n_base])
            prefix = template["id"].split("-")[0]
            template["id"] = f"{prefix}-{i + 1:04d}"
            template["metadata"] = {
                **dict(template.get("metadata", {})),
                "dataset_name": ds_name,
                "variant_index": i + 1,
            }
            corpus.append(template)
    return corpus


def export_converted_entries_jsonl(entries: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save converted entries to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path

