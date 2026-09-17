from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.prompts import EVIDENCE_TEMPLATE, QUERY_TEMPLATE, REPO_MAP_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from the CodeDoctor agent."""

    question: str
    answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool = True
    model_used: str = ""


class CodeDoctorAgent:
    """LLM-powered code intelligence agent with retrieval-augmented generation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._llm_available = False
        try:
            import importlib.util

            self._llm_available = importlib.util.find_spec("litellm") is not None
        except ImportError:
            logger.warning("litellm not installed. Agent will return evidence only.")

    def _format_evidence(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into evidence blocks."""
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(
                EVIDENCE_TEMPLATE.format(
                    index=i,
                    file_path=chunk.get("file_path", "unknown"),
                    start_line=chunk.get("start_line", 0),
                    end_line=chunk.get("end_line", 0),
                    type=chunk.get("type", "UNKNOWN"),
                    name=chunk.get("name", "unknown"),
                    content=chunk.get("content", chunk.get("snippet", "")),
                )
            )
        return "\n".join(blocks)

    def ask(
        self,
        question: str,
        evidence_chunks: list[dict[str, Any]],
        repo_map: str = "",
    ) -> AgentResponse:
        """Answer a question using retrieved code evidence.

        If litellm is available, sends the evidence to an LLM for synthesis.
        Otherwise returns the formatted evidence for the user to review.
        """
        evidence_text = self._format_evidence(evidence_chunks)

        if not self._llm_available:
            return AgentResponse(
                question=question,
                answer=f"LLM not available (install litellm). Here is the retrieved evidence:\n\n{evidence_text}",
                evidence=evidence_chunks,
                grounded=True,
                model_used="none",
            )

        try:
            import litellm

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            if repo_map:
                messages.append(
                    {
                        "role": "system",
                        "content": REPO_MAP_PROMPT.format(repo_map=repo_map),
                    }
                )

            user_message = QUERY_TEMPLATE.format(
                question=question,
                evidence_blocks=evidence_text,
            )
            messages.append({"role": "user", "content": user_message})

            kwargs = {"model": self.model, "messages": messages}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = litellm.completion(**kwargs)
            answer = response.choices[0].message.content

            return AgentResponse(
                question=question,
                answer=answer,
                evidence=evidence_chunks,
                grounded=True,
                model_used=self.model,
            )
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return AgentResponse(
                question=question,
                answer=f"LLM call failed ({e}). Here is the retrieved evidence:\n\n{evidence_text}",
                evidence=evidence_chunks,
                grounded=True,
                model_used=self.model,
            )

    def explain_symbol(
        self,
        symbol_name: str,
        definition_chunk: dict[str, Any],
        caller_chunks: list[dict[str, Any]],
        callee_chunks: list[dict[str, Any]],
    ) -> AgentResponse:
        """Explain a symbol by analyzing its definition, callers, and callees."""
        all_evidence = [definition_chunk, *caller_chunks, *callee_chunks]
        question = (
            f"Explain the purpose and behavior of '{symbol_name}'. "
            f"Describe what it does, who calls it ({len(caller_chunks)} callers), "
            f"and what it calls ({len(callee_chunks)} callees). "
            f"Explain the data flow and any important patterns."
        )
        return self.ask(question, all_evidence)

    def review_code(
        self,
        code_chunks: list[dict[str, Any]],
        focus: str = "general",
    ) -> AgentResponse:
        """Review code chunks for issues, patterns, and improvements."""
        question = (
            f"Review the following code with a focus on: {focus}. "
            f"Identify potential bugs, security issues, performance problems, "
            f"and suggest improvements. Be specific with line references."
        )
        return self.ask(question, code_chunks)
