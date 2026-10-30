"""Typed Python SDK for MicroLLM (`POST /v1/systemone`), compatible with TypeSafe & poorjev idioms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import functools
from typing import Any, Callable, Mapping, Sequence, Type, Union
import httpx


@dataclass(frozen=True)
class Noul:
    """Typed helper for a Yes/No (`noul`) question."""

    instructions: str | None = None
    true_description: str | None = None
    false_description: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul"}
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        criteria: dict[str, str] = {}
        if self.true_description is not None:
            criteria["true"] = self.true_description
        if self.false_description is not None:
            criteria["false"] = self.false_description
        if criteria:
            payload["criteria"] = criteria
        return payload


@dataclass(frozen=True)
class Choice:
    """Typed helper for a mutually-exclusive `choice` question (2..255 options)."""

    options: Union[Sequence[str], Mapping[str, str | None]]
    instructions: str = "Select the most appropriate option."

    @classmethod
    def from_enum(
        cls,
        enum_cls: Type[Enum],
        instructions: str = "Select the most appropriate option.",
    ) -> "Choice":
        """Build a typed Choice question directly from a Python Enum class (IDEA-016)."""
        mapping = {
            member.name: (str(member.value) if member.value is not None else None)
            for member in enum_cls
        }
        return cls(options=mapping, instructions=instructions)

    def to_payload(self) -> dict[str, Any]:
        if isinstance(self.options, Mapping):
            criteria = dict(self.options)
        else:
            criteria = {str(opt): None for opt in self.options}
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": criteria,
        }


@dataclass(frozen=True)
class Score:
    """Typed helper for an ordinal `score` rubric question (2..10 levels)."""

    levels: Sequence[Any]
    instructions: str = "Rate on the given ordered scale."

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }


@dataclass(frozen=True)
class Multi:
    """Typed helper for an independent multi-label `multi` question (2..255 labels)."""

    options: Union[Sequence[str], Mapping[str, str | None]]
    instructions: str = "Select all applicable labels."
    threshold: float = 0.5

    @classmethod
    def from_enum(
        cls,
        enum_cls: Type[Enum],
        instructions: str = "Select all applicable labels.",
        threshold: float = 0.5,
    ) -> "Multi":
        """Build a typed Multi question directly from a Python Enum class (IDEA-016)."""
        mapping = {
            member.name: (str(member.value) if member.value is not None else None)
            for member in enum_cls
        }
        return cls(options=mapping, instructions=instructions, threshold=threshold)

    def to_payload(self) -> dict[str, Any]:
        if isinstance(self.options, Mapping):
            criteria = dict(self.options)
        else:
            criteria = {str(opt): None for opt in self.options}
        return {
            "type": "multi",
            "instructions": self.instructions,
            "criteria": criteria,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class NoulResult:
    """Parsed answer for a `noul` question."""

    type: str
    probability: float
    threshold: float = 0.5

    @property
    def value(self) -> bool:
        return self.probability >= self.threshold


@dataclass(frozen=True)
class ChoiceResult:
    """Parsed answer for a `choice` question."""

    type: str
    value: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreResult:
    """Parsed answer for a `score` question."""

    type: str
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float

    @property
    def value(self) -> Any:
        best_idx = max(self.probabilities, key=lambda k: self.probabilities[k])
        return self.legend[best_idx]


@dataclass(frozen=True)
class MultiResult:
    """Parsed answer for an independent multi-label `multi` question."""

    type: str
    labels: list[str]
    probabilities: dict[str, float]
    confidence: float

    @property
    def value(self) -> list[str]:
        return self.labels


ParsedResult = Union[NoulResult, ChoiceResult, ScoreResult, MultiResult]


class MicroJevClient:
    """Typed System-One client for local or remote `/v1/systemone` endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "local-jev-key",
        model: str = "jev-latest",
        http_client: Any | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client
        self.multilingual_strategy = multilingual_strategy
        self.neural_model = neural_model

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Union[Noul, Choice, Score, Multi, dict[str, Any]]],
        model: str | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
    ) -> dict[str, ParsedResult]:
        """Send a batched System-One request and return strongly-typed answer objects."""
        serialized_questions: dict[str, Any] = {}
        for qid, q_spec in questions.items():
            if hasattr(q_spec, "to_payload"):
                serialized_questions[qid] = q_spec.to_payload()  # type: ignore[union-attr]
            else:
                serialized_questions[qid] = dict(q_spec)  # type: ignore[arg-type]

        payload: dict[str, Any] = {
            "model": model or self.model,
            "state": state,
            "questions": serialized_questions,
        }
        eff_strat = multilingual_strategy if multilingual_strategy is not None else self.multilingual_strategy
        eff_neural = neural_model if neural_model is not None else self.neural_model
        if eff_strat is not None:
            payload["multilingual_strategy"] = eff_strat
        if eff_neural is not None:
            payload["neural_model"] = eff_neural

        headers = {"Authorization": f"Bearer {self.api_key}"}

        if self._http is not None:
            resp = self._http.post("/v1/systemone", json=payload, headers=headers)
        else:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                resp = client.post("/v1/systemone", json=payload, headers=headers)

        resp.raise_for_status()
        body = resp.json()
        raw_answers = body["answers"]

        parsed: dict[str, ParsedResult] = {}
        for qid, ans in raw_answers.items():
            q_type = ans["type"]
            if q_type == "noul":
                parsed[qid] = NoulResult(type="noul", probability=float(ans["noul"]))
            elif q_type == "choice":
                parsed[qid] = ChoiceResult(
                    type="choice",
                    value=str(ans["choice"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "score":
                parsed[qid] = ScoreResult(
                    type="score",
                    score=float(ans["score"]),
                    legend=dict(ans["legend"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "multi":
                parsed[qid] = MultiResult(
                    type="multi",
                    labels=list(ans["labels"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
        return parsed

    def evaluate_choice(
        self,
        state: Any,
        options: Union[Sequence[str], Mapping[str, str | None]],
        instructions: str = "Select the most appropriate option.",
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> ChoiceResult:
        """Typed helper to evaluate a single Choice question."""
        res = self.ask(
            state=state,
            questions={"choice": Choice(options=options, instructions=instructions)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["choice"]
        assert isinstance(ans, ChoiceResult)
        return ans

    def evaluate_score(
        self,
        state: Any,
        levels: Sequence[Any],
        instructions: str = "Rate on the given ordered scale.",
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> ScoreResult:
        """Typed helper to evaluate a single Score rubric question."""
        res = self.ask(
            state=state,
            questions={"score": Score(levels=levels, instructions=instructions)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["score"]
        assert isinstance(ans, ScoreResult)
        return ans

    def evaluate_multi(
        self,
        state: Any,
        options: Union[Sequence[str], Mapping[str, str | None]],
        instructions: str = "Select all applicable labels.",
        threshold: float = 0.5,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> MultiResult:
        """Typed helper to evaluate a single Multi-label question."""
        res = self.ask(
            state=state,
            questions={"multi": Multi(options=options, instructions=instructions, threshold=threshold)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["multi"]
        assert isinstance(ans, MultiResult)
        return ans

    def evaluate_noul(
        self,
        state: Any,
        instructions: str | None = None,
        true_description: str | None = None,
        false_description: str | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> NoulResult:
        """Typed helper to evaluate a single Yes/No proposition."""
        res = self.ask(
            state=state,
            questions={
                "noul": Noul(
                    instructions=instructions,
                    true_description=true_description,
                    false_description=false_description,
                )
            },
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["noul"]
        assert isinstance(ans, NoulResult)
        return ans

    def guardrail(
        self,
        state: Any,
        context: Any | None = None,
        threshold: float = 0.55,
        custom_rubrics: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate AI safety guardrails (`POST /v1/guardrail`) via the typed SDK (IDEA-017/018)."""
        payload: dict[str, Any] = {"state": state, "threshold": float(threshold)}
        if context is not None:
            payload["context"] = context
        if custom_rubrics is not None:
            payload["custom_rubrics"] = dict(custom_rubrics)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._http is not None:
            resp = self._http.post("/v1/guardrail", json=payload, headers=headers)
        else:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                resp = client.post("/v1/guardrail", json=payload, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())

    def batch(
        self,
        items: Sequence[Any],
        questions: Mapping[str, Union[Noul, Choice, Score, Multi, dict[str, Any]]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of states (`POST /v1/batch`) via the typed SDK (IDEA-013/017)."""
        serialized_questions: dict[str, Any] = {}
        for qid, q_spec in questions.items():
            if hasattr(q_spec, "to_payload"):
                serialized_questions[qid] = q_spec.to_payload()  # type: ignore[union-attr]
            else:
                serialized_questions[qid] = dict(q_spec)  # type: ignore[arg-type]
        payload = {
            "items": list(items),
            "questions": serialized_questions,
            "model": model or self.model,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._http is not None:
            resp = self._http.post("/v1/batch", json=payload, headers=headers)
        else:
            with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
                resp = client.post("/v1/batch", json=payload, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())

    def papers(
        self,
        pillar: str | None = None,
        query: str | None = None,
        arxiv_id: str | None = None,
    ) -> dict[str, Any]:
        """Query the curated Jev, RLCD, System-1 & Calibrated LLM academic catalog (`GET /v1/papers`)."""
        params: dict[str, str] = {}
        if pillar is not None:
            params["pillar"] = pillar
        if query is not None:
            params["q"] = query
        if arxiv_id is not None:
            params["arxiv_id"] = arxiv_id
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._http is not None:
            resp = self._http.get("/v1/papers", params=params, headers=headers)
        else:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                resp = client.get("/v1/papers", params=params, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())


class AsyncMicroJevClient:
    """Asynchronous Typed System-One SDK for non-blocking `asyncio` workflows (IDEA-015)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "local-jev-key",
        model: str = "jev-latest",
        http_client: Any | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client
        self.multilingual_strategy = multilingual_strategy
        self.neural_model = neural_model

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Union[Noul, Choice, Score, Multi, dict[str, Any]]],
        model: str | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
    ) -> dict[str, ParsedResult]:
        """Asynchronously send a batched System-One request and return strongly-typed answers."""
        serialized_questions: dict[str, Any] = {}
        for qid, q_spec in questions.items():
            if hasattr(q_spec, "to_payload"):
                serialized_questions[qid] = q_spec.to_payload()  # type: ignore[union-attr]
            else:
                serialized_questions[qid] = dict(q_spec)  # type: ignore[arg-type]

        payload: dict[str, Any] = {
            "model": model or self.model,
            "state": state,
            "questions": serialized_questions,
        }
        eff_strat = multilingual_strategy if multilingual_strategy is not None else self.multilingual_strategy
        eff_neural = neural_model if neural_model is not None else self.neural_model
        if eff_strat is not None:
            payload["multilingual_strategy"] = eff_strat
        if eff_neural is not None:
            payload["neural_model"] = eff_neural

        headers = {"Authorization": f"Bearer {self.api_key}"}

        if self._http is not None:
            maybe_coro = self._http.post("/v1/systemone", json=payload, headers=headers)
            resp = await maybe_coro if hasattr(maybe_coro, "__await__") else maybe_coro
        else:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                resp = await client.post("/v1/systemone", json=payload, headers=headers)

        resp.raise_for_status()
        raw_answers = resp.json()["answers"]
        parsed: dict[str, ParsedResult] = {}
        for qid, ans in raw_answers.items():
            q_type = ans["type"]
            if q_type == "noul":
                parsed[qid] = NoulResult(type="noul", probability=float(ans["noul"]))
            elif q_type == "choice":
                parsed[qid] = ChoiceResult(
                    type="choice",
                    value=str(ans["choice"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "score":
                parsed[qid] = ScoreResult(
                    type="score",
                    score=float(ans["score"]),
                    legend=dict(ans["legend"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
            elif q_type == "multi":
                parsed[qid] = MultiResult(
                    type="multi",
                    labels=list(ans["labels"]),
                    probabilities=dict(ans["probabilities"]),
                    confidence=float(ans["confidence"]),
                )
        return parsed

    async def evaluate_choice(
        self,
        state: Any,
        options: Union[Sequence[str], Mapping[str, str | None]],
        instructions: str = "Select the most appropriate option.",
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> ChoiceResult:
        """Asynchronously evaluate a single Choice question."""
        res = await self.ask(
            state=state,
            questions={"choice": Choice(options=options, instructions=instructions)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["choice"]
        assert isinstance(ans, ChoiceResult)
        return ans

    async def evaluate_score(
        self,
        state: Any,
        levels: Sequence[Any],
        instructions: str = "Rate on the given ordered scale.",
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> ScoreResult:
        """Asynchronously evaluate a single Score rubric question."""
        res = await self.ask(
            state=state,
            questions={"score": Score(levels=levels, instructions=instructions)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["score"]
        assert isinstance(ans, ScoreResult)
        return ans

    async def evaluate_multi(
        self,
        state: Any,
        options: Union[Sequence[str], Mapping[str, str | None]],
        instructions: str = "Select all applicable labels.",
        threshold: float = 0.5,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> MultiResult:
        """Asynchronously evaluate a single Multi-label question."""
        res = await self.ask(
            state=state,
            questions={"multi": Multi(options=options, instructions=instructions, threshold=threshold)},
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["multi"]
        assert isinstance(ans, MultiResult)
        return ans

    async def evaluate_noul(
        self,
        state: Any,
        instructions: str | None = None,
        true_description: str | None = None,
        false_description: str | None = None,
        multilingual_strategy: str | None = None,
        neural_model: str | None = None,
        model: str | None = None,
    ) -> NoulResult:
        """Asynchronously evaluate a single Yes/No proposition."""
        res = await self.ask(
            state=state,
            questions={
                "noul": Noul(
                    instructions=instructions,
                    true_description=true_description,
                    false_description=false_description,
                )
            },
            multilingual_strategy=multilingual_strategy,
            neural_model=neural_model,
            model=model,
        )
        ans = res["noul"]
        assert isinstance(ans, NoulResult)
        return ans

    async def papers(
        self,
        pillar: str | None = None,
        query: str | None = None,
        arxiv_id: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously query the Jev & System-1 academic catalog (`GET /v1/papers`)."""
        params: dict[str, str] = {}
        if pillar is not None:
            params["pillar"] = pillar
        if query is not None:
            params["q"] = query
        if arxiv_id is not None:
            params["arxiv_id"] = arxiv_id
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._http is not None:
            maybe_coro = self._http.get("/v1/papers", params=params, headers=headers)
            resp = await maybe_coro if hasattr(maybe_coro, "__await__") else maybe_coro
        else:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                resp = await client.get("/v1/papers", params=params, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())


def system_one_router(
    client: MicroJevClient,
    options: Union[Type[Enum], Sequence[str], Mapping[str, str | None]],
    instructions: str = "Route the input state to the best handler.",
    model: str | None = None,
    multilingual_strategy: str | None = None,
    neural_model: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator turning a function into a calibrated System-1 router (IDEA-017)."""
    if isinstance(options, type) and issubclass(options, Enum):
        choice_q = Choice.from_enum(options, instructions=instructions)
    else:
        choice_q = Choice(options=options, instructions=instructions)

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def _wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            res = client.ask(
                state=state,
                questions={"route": choice_q},
                model=model,
                multilingual_strategy=multilingual_strategy,
                neural_model=neural_model,
            )
            route_ans = res["route"]
            assert isinstance(route_ans, ChoiceResult)
            return func(
                state,
                *args,
                decision=route_ans.value,
                confidence=route_ans.confidence,
                **kwargs,
            )

        return _wrapper

    return _decorator
