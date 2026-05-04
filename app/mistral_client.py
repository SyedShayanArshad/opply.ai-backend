from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
import anyio


def _load_env() -> None:
    load_dotenv()


class MistralLLM:
    def __init__(self) -> None:
        _load_env()
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.model = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

        self._client = None
        if self.api_key:
            try:
                from mistralai import Mistral  # type: ignore

                self._client = Mistral(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def json_extract(self, *, system: str, user: str, schema_hint: str) -> dict[str, Any]:
        """Returns parsed JSON dict; raises on failure."""
        if not self._client:
            raise RuntimeError("Mistral client not configured")

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{user}\n\nReturn ONLY valid JSON.\nSchema hint:\n{schema_hint}",
            },
        ]

        # Run the SDK's sync call off the event loop.
        resp = await anyio.to_thread.run_sync(
            lambda: self._client.chat.complete(model=self.model, messages=messages)
        )
        content = resp.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Mistral response")
        
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model did not return valid JSON: {e}\nContent was: {content}") from e
