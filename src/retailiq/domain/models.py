"""Transport-agnostic result objects."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retailiq.core.enums import Route


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TraceStep(BaseModel):
    """One decision the agent made, in order.

    The trace is a product feature, not debug output: it is what turns the
    agent from a black box into something a store manager (or an auditor)
    can see the reasoning of.
    """

    model_config = ConfigDict(frozen=True)

    step: str = Field(description="Node that ran, e.g. 'grade_documents'.")
    detail: str = Field(description="Human-readable summary of what it decided.")

    def __str__(self) -> str:
        return f"{self.step}: {self.detail}"


class RetrievedChunk(BaseModel):
    """A knowledge-base chunk that survived relevance grading."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Originating document, e.g. 'returns_policy.md'.")
    topic: str = Field(default="", description="Document stem, used for filtering.")
    content: str = Field(description="Chunk text supplied to the generator.")

    def render(self) -> str:
        """Format for inclusion in a prompt, tagged with its source."""
        return f"[{self.source}]\n{self.content}"


class QueryResult(BaseModel):
    """The full outcome of one question, including how it was reached."""

    question: str
    answer: str
    route: Route = Route.VECTORSTORE
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    rewrites: int = Field(default=0, description="Times the query was reformulated.")
    grounded: bool = Field(default=True, description="Passed the hallucination gate.")
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def context(self) -> str:
        """The exact context block the generator saw."""
        if not self.chunks:
            return "(no context retrieved)"
        return "\n\n---\n\n".join(chunk.render() for chunk in self.chunks)

    @property
    def sources(self) -> list[str]:
        """Unique source documents behind the answer, in first-seen order."""
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.source, None)
        return list(seen)


class ChatTurn(BaseModel):
    """One message in a conversation.

    Assistant turns carry the provenance fields alongside the text, so a
    reloaded conversation still shows *why* each answer was given rather than
    degrading into a plain transcript.
    """

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=_now)

    # Assistant-only. Left unset on user turns.
    sources: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    route: Route | None = None
    grounded: bool | None = None
    latency_ms: int | None = None


class Conversation(BaseModel):
    """A named, persisted sequence of turns."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New chat"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    turns: list[ChatTurn] = Field(default_factory=list)

    def add(self, turn: ChatTurn) -> None:
        self.turns.append(turn)
        self.updated_at = _now()
        # Name the thread from its opening question, the way a chat client
        # does — a list of "New chat" entries is unusable.
        if self.title == "New chat" and turn.role == "user":
            self.title = self._derive_title(turn.content)

    @staticmethod
    def _derive_title(text: str, limit: int = 48) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    def history_before_last(self) -> list[ChatTurn]:
        """Prior turns, excluding the user message being answered right now."""
        return self.turns[:-1] if self.turns else []

    @property
    def is_empty(self) -> bool:
        return not self.turns


class EvaluationCase(BaseModel):
    """One golden question from the evaluation dataset."""

    question: str
    expected_source: str = Field(description="Document that should be retrieved.")
    must_include: list[str] = Field(
        default_factory=list, description="Substrings the answer must contain."
    )


class EvaluationRowResult(BaseModel):
    """Per-question scoring outcome."""

    question: str
    retrieved_expected_source: bool
    keywords_present: bool
    faithful: bool
    relevant: bool
    answer: str = ""

    @property
    def passed(self) -> bool:
        """A row passes only if every metric passes."""
        return all(
            (
                self.retrieved_expected_source,
                self.keywords_present,
                self.faithful,
                self.relevant,
            )
        )


class EvaluationReport(BaseModel):
    """Aggregate quality metrics over the evaluation dataset."""

    rows: list[EvaluationRowResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total(self) -> int:
        return len(self.rows)

    def _rate(self, attribute: str) -> float:
        if not self.rows:
            return 0.0
        return sum(bool(getattr(row, attribute)) for row in self.rows) / len(self.rows)

    @property
    def retrieval_hit_rate(self) -> float:
        """Did the expected source document reach the context?"""
        return self._rate("retrieved_expected_source")

    @property
    def keyword_recall(self) -> float:
        """Did the answer state the required fact?"""
        return self._rate("keywords_present")

    @property
    def faithfulness(self) -> float:
        """LLM-as-judge: is every claim supported by the context?"""
        return self._rate("faithful")

    @property
    def answer_relevance(self) -> float:
        """LLM-as-judge: does the answer address the question?"""
        return self._rate("relevant")

    def summary(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 4),
            "keyword_recall": round(self.keyword_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
        }
