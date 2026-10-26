"""app/llm_client.py

Thin wrapper around any OpenAI-compatible chat API (OpenAI, Groq, Hugging Face).
- Loads versioned prompts from prompts/prompts.yaml
- Structured output validated against a Pydantic model
- Retries transient API errors and invalid outputs
- Never logs patient content (prompt name, version, latency only)
"""
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Protocol, Type, TypeVar

import yaml
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompts.yaml"
# APITimeoutError is a subclass of APIConnectionError
TRANSIENT_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLM(Protocol):
    """What the rest of the code depends on. LLMClient and FakeLLMClient both fit."""

    async def complete_structured(
        self, prompt_name: str, variables: dict[str, str], output_model: Type[T],
        temperature: float | None = 0.0,
    ) -> T: ...

    async def complete_text(
        self, prompt_name: str, variables: dict[str, str], temperature: float | None = 0.3,
    ) -> str: ...


class LLMClient:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise RuntimeError("No LLM API key. Set LLM_API_KEY (or GROQ_API_KEY) in .env")
        if not settings.llm_model:
            raise RuntimeError("No model. Set LLM_MODEL in .env")
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # we retry ourselves so every attempt is logged
        )
        self._prompts = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))

    # ---------- prompts ----------
    def _load_prompt(self, name: str) -> tuple[str, str]:
        entry = self._prompts[name]
        version = entry["active"]
        return version, entry["versions"][version]

    @staticmethod
    def _render(template: str, variables: dict[str, str]) -> str:
        # Single pass, so text inside a variable is never re-substituted.
        return PLACEHOLDER.sub(lambda m: variables.get(m.group(1), m.group(0)), template)

    # ---------- public API ----------
    async def complete_structured(
        self,
        prompt_name: str,
        variables: dict[str, str],
        output_model: Type[T],
        temperature: float | None = 0.0,
    ) -> T:
        version, template = self._load_prompt(prompt_name)
        prompt = self._render(template, variables)
        kwargs = {} if temperature is None else {"temperature": temperature}

        async def call() -> T:
            if settings.llm_structured_mode == "schema":
                resp = await self._client.chat.completions.parse(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=output_model,
                    **kwargs,
                )
                result = resp.choices[0].message.parsed
                if result is None:  # refusal or empty output
                    raise ValueError("no parsed output")
                return result

            schema = json.dumps(output_model.model_json_schema())
            resp = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[{
                    "role": "user",
                    "content": prompt
                    + "\n\nReply with ONLY a JSON object (no markdown) matching this JSON schema. "
                    "Include every field; use null or [] when there is nothing to report.\n"
                    + schema,
                }],
                response_format={"type": "json_object"},
                **kwargs,
            )
            return output_model.model_validate_json(resp.choices[0].message.content or "")

        return await self._with_retries(prompt_name, version, call)

    async def complete_text(
        self,
        prompt_name: str,
        variables: dict[str, str],
        temperature: float | None = 0.3,
    ) -> str:
        version, template = self._load_prompt(prompt_name)
        prompt = self._render(template, variables)
        kwargs = {} if temperature is None else {"temperature": temperature}

        async def call() -> str:
            resp = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("empty output")
            return text

        return await self._with_retries(prompt_name, version, call)

    # ---------- internals ----------
    async def _with_retries(self, name: str, version: str, call: Callable[[], Awaitable[T]]) -> T:
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            start = time.perf_counter()
            try:
                result = await call()
                self._log(name, version, start, attempt, "ok")
                return result
            except TRANSIENT_ERRORS as e:
                last_error = e
                self._log(name, version, start, attempt, f"transient:{type(e).__name__}")
                await asyncio.sleep(2**attempt)  # 1s, 2s, 4s...
            except (ValueError, LengthFinishReasonError) as e:  # includes pydantic ValidationError
                last_error = e
                self._log(name, version, start, attempt, "invalid_output")
            except APIStatusError as e:  # 400/401/404...: retrying will not help
                self._log(name, version, start, attempt, f"api_error:{e.status_code}")
                raise LLMError(f"{name}@{version} API error {e.status_code}") from e
        # Only the exception TYPE goes into the message: validation errors can echo patient text.
        raise LLMError(f"{name}@{version} failed: {type(last_error).__name__}")

    @staticmethod
    def _log(name: str, version: str, start: float, attempt: int, status: str) -> None:
        logger.info(
            "llm_call prompt=%s version=%s attempt=%d status=%s latency_ms=%d",
            name, version, attempt, status, (time.perf_counter() - start) * 1000,
        )


class FakeLLMClient:
    """Drop-in replacement for tests. Canned outputs keyed by prompt name."""

    def __init__(self, structured: dict[str, BaseModel] | None = None,
                 text: dict[str, str] | None = None) -> None:
        self.structured = structured or {}
        self.text = text or {}
        self.calls: list[tuple[str, dict]] = []

    async def complete_structured(self, prompt_name, variables, output_model, temperature=0.0):
        self.calls.append((prompt_name, variables))
        return self.structured[prompt_name]

    async def complete_text(self, prompt_name, variables, temperature=0.3):
        self.calls.append((prompt_name, variables))
        return self.text[prompt_name]
