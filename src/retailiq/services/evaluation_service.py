"""Evaluation harness — makes answer quality a number rather than a vibe.

Four metrics over a golden dataset, spanning both halves of the pipeline:

* **Retrieval hit-rate** — did the expected source document reach the context?
  Isolates retrieval failure from generation failure. If this drops, the
  problem is chunking or embeddings, not the prompt.
* **Keyword recall** — does the answer state the required fact ("30 days")?
  Deterministic and cheap; catches fluent answers that omit the number.
* **Faithfulness** (LLM judge) — is every claim supported by the context?
  The offline analogue of the runtime hallucination gate.
* **Answer relevance** (LLM judge) — does it address the question?

Implemented without RAGAS on purpose: RAGAS pins its own provider stack,
whereas these judges run against whichever provider is configured, so the
harness works on a free Groq key or fully offline via Ollama.
"""

from __future__ import annotations

import json

from retailiq.agent import prompts
from retailiq.core.exceptions import EvaluationError
from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.domain.models import EvaluationCase, EvaluationReport, EvaluationRowResult
from retailiq.llm.factory import get_chat_model
from retailiq.services.rag_service import RAGService

logger = get_logger(__name__)


class EvaluationService:
    """Scores the assistant against the golden question set."""

    def __init__(
        self, settings: Settings | None = None, rag_service: RAGService | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._rag = rag_service or RAGService(self._settings)

    def load_cases(self) -> list[EvaluationCase]:
        """Read and validate the evaluation dataset."""
        path = self._settings.paths.eval_dataset
        if not path.exists():
            raise EvaluationError(f"Evaluation dataset not found: {path}", path=str(path))

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Evaluation dataset is not valid JSON: {exc}") from exc

        if not isinstance(raw, list) or not raw:
            raise EvaluationError("Evaluation dataset must be a non-empty JSON array.")

        try:
            return [EvaluationCase(**row) for row in raw]
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Invalid evaluation case: {exc}") from exc

    def run(self) -> EvaluationReport:
        """Execute every case and aggregate the results."""
        cases = self.load_cases()
        logger.info("Starting evaluation", extra={"cases": len(cases)})

        rows = [self._score(case) for case in cases]
        report = EvaluationReport(rows=rows)

        logger.info("Evaluation complete", extra=report.summary())
        return report

    # -- internals ---------------------------------------------------------
    def _score(self, case: EvaluationCase) -> EvaluationRowResult:
        result = self._rag.answer(case.question)
        answer, context = result.answer, result.context

        return EvaluationRowResult(
            question=case.question,
            answer=answer,
            retrieved_expected_source=case.expected_source in context,
            keywords_present=all(token.lower() in answer.lower() for token in case.must_include),
            faithful=self._judge(
                prompts.FAITHFULNESS_JUDGE_SYSTEM,
                f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
            ),
            relevant=self._judge(
                prompts.RELEVANCE_JUDGE_SYSTEM,
                f"QUESTION: {case.question}\n\nANSWER:\n{answer}",
            ),
        )

    def _judge(self, system: str, user: str) -> bool:
        """Binary LLM-as-judge call. Temperature 0 keeps verdicts stable."""
        from langchain_core.messages import HumanMessage, SystemMessage

        response = get_chat_model(temperature=0.0, settings=self._settings).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = str(response.content).strip()
        return bool(text) and text.split()[0].strip("\"'.,:;").lower() == "yes"
