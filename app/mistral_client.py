"""
mistral_client.py — LangChain-backed Mistral LLM client

Replaces the direct mistralai SDK with LangChain's ChatMistralAI so that
all LLM interactions go through the LangChain abstraction layer.
Same external interface (json_extract / generate_text) — zero changes needed
in callers (extract.py, profile_summary.py, record_explanation.py).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anyio
from dotenv import load_dotenv

logger = logging.getLogger("app.mistral_client")


def _load_env() -> None:
    load_dotenv()


class MistralLLM:
    """LangChain ChatMistralAI wrapper with the same interface as the old client."""

    def __init__(self) -> None:
        _load_env()
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.model = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()
        self._llm = None

        if self.api_key:
            try:
                from langchain_mistralai import ChatMistralAI  # type: ignore

                self._llm = ChatMistralAI(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.0,
                )
                logger.info("LangChain ChatMistralAI ready (model: %s).", self.model)
            except Exception as exc:
                logger.warning("LangChain ChatMistralAI init failed: %s", exc)
                self._llm = None

    @property
    def available(self) -> bool:
        return self._llm is not None

    # ── Core method ────────────────────────────────────────────────────────────

    async def json_extract(
        self, *, system: str, user: str, schema_hint: str
    ) -> dict[str, Any]:
        """
        Send a chat completion and parse the response as JSON.
        Uses LangChain's ChatPromptTemplate + ChatMistralAI chain.
        Raises RuntimeError on failure so callers can fall back gracefully.
        """
        if not self._llm:
            raise RuntimeError("LangChain Mistral client not configured.")

        from langchain_core.prompts import ChatPromptTemplate  # type: ignore

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                (
                    "human",
                    "{user_input}\n\nReturn ONLY valid JSON.\nSchema hint:\n{schema_hint}",
                ),
            ]
        )
        chain = prompt | self._llm

        response = await anyio.to_thread.run_sync(
            lambda: chain.invoke(
                {"system": system, "user_input": user, "schema_hint": schema_hint}
            )
        )

        content: str = (
            response.content if hasattr(response, "content") else str(response)
        )
        content = content.strip()

        # Strip markdown code fences that some models add
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Model did not return valid JSON: {exc}\nRaw content: {content}"
            ) from exc

    async def generate_text(self, *, system: str, user: str) -> str:
        """
        Send a chat completion and return the plain-text response.
        """
        if not self._llm:
            raise RuntimeError("LangChain Mistral client not configured.")

        from langchain_core.prompts import ChatPromptTemplate  # type: ignore

        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", "{user_input}")]
        )
        chain = prompt | self._llm

        response = await anyio.to_thread.run_sync(
            lambda: chain.invoke({"system": system, "user_input": user})
        )

        return (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )
