
from __future__ import annotations
import asyncio
from collections.abc import Sequence
from typing import Final, Literal, Protocol, runtime_checkable
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from ft.config import get_settings


DEFAULT_TEMPERATURE: Final[float] = 0.2
DEFAULT_QUESTIONS_PER_CHUNK: Final[int] = 3

SYSTEM_PROMPT_TEMPLATE: Final[str] = """
You are a senior software engineer and technical educator building a Supervised Fine-Tuning (SFT) dataset.
Given a section title and source excerpt, generate {num_pairs} distinct, high-signal instruction-output pairs.

Intent Diversity (distribute pairs across these categories):
- "implementation": Practical how-to tasks, API usage, or runnable code snippets demonstrating the concept.
- "debugging": Common developer mistakes, edge cases, error handling, or troubleshooting scenarios.
- "architecture": Design trade-offs, internal behavior, or when to choose one approach over another.

Strict Quality Rules:
1. Ground every fact, signature, and behavior strictly in the provided excerpt. Never invent APIs or external assumptions.
2. Write self-contained instructions: do NOT reference "the text", "the excerpt", or "the document above".
3. Write direct, authoritative outputs: omit conversational filler (no "Sure!", "Here is...", or "Based on the text").
4. When providing code, use clean, idiomatic syntax inside fenced Markdown blocks.
5. Write both instructions and outputs in the exact same language as the source excerpt.
""".strip()

USER_PROMPT_TEMPLATE: Final[str] = """
Section: {section_title}

Content:
\"\"\"
{content}
\"\"\"
""".strip()


@runtime_checkable
class ChunkProtocol(Protocol):
    """
    Structural contract decoupling the synthesizer from any specific ingestor class.
    """

    @property
    def chunk_id(self) -> str: ...

    @property
    def section_title(self) -> str: ...

    @property
    def content(self) -> str: ...


class QAPair(BaseModel):

    category: Literal["implementation", "debugging", "architecture"]
    instruction: str = Field(..., min_length=5)
    output: str = Field(..., min_length=10)

    model_config = ConfigDict(
        frozen=True
    )


class _ChunkSynthesisResponse(BaseModel):

    pairs: list[QAPair]


class ChatMessage(BaseModel):

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    model_config = ConfigDict(
        frozen=True    
    )


class SFTRecord(BaseModel):

    chunk_id: str = Field(..., min_length=1)
    category: Literal["implementation", "debugging", "architecture"]
    messages: list[ChatMessage] = Field(..., min_length=2, max_length=2)

    model_config = ConfigDict(
        frozen=True
    )


class DatasetSynthesizer:
    """
    Synthesizes structured SFT training records from text chunks using an OpenAI-compatible API.

    Operates asynchronously with bounded concurrency via `asyncio.Semaphore` and enforces
    Pydantic structured outputs. Connection credentials may be injected explicitly at
    instantiation or resolved lazily from `ft.config.get_settings()`.

    Attributes:
        model (Final[str]): Target LLM identifier used for synthetic generation.
        max_concurrency (Final[int]): Maximum simultaneous in-flight API requests.
        questions_per_chunk (Final[int]): Target number of QA pairs generated per chunk.
        temperature (Final[float]): Sampling temperature applied during generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        verify_ssl: bool | None = None,
        max_concurrency: int | None = None,
        questions_per_chunk: int = DEFAULT_QUESTIONS_PER_CHUNK,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """
        Initializes the DatasetSynthesizer with explicit or fallback configuration.

        Args:
            api_key (str | None, optional): API key for the OpenAI-compatible server.
                Falls back to `Settings.AI_API_KEY` if omitted. Defaults to None.
            base_url (str | None, optional): Endpoint base URL (e.g., local vLLM/Ollama).
                Falls back to `Settings.AI_BASE_URL` if omitted. Defaults to None.
            model (str | None, optional): Model name to invoke. Falls back to
                `Settings.AI_MODEL` if omitted. Defaults to None.
            verify_ssl (bool | None, optional): Whether to verify TLS certificates.
                Falls back to `Settings.AI_VERIFY_SSL` if omitted. Defaults to None.
            max_concurrency (int | None, optional): Maximum concurrent async requests.
                Falls back to `Settings.AI_MAX_CONCURRENCY` if omitted. Defaults to None.
            questions_per_chunk (int, optional): Number of QA pairs requested per chunk.
                Must be positive. Defaults to `DEFAULT_QUESTIONS_PER_CHUNK`.
            temperature (float, optional): Sampling temperature in `[0.0, 2.0]`.
                Defaults to `DEFAULT_TEMPERATURE`.

        Raises:
            ValueError: If `questions_per_chunk < 1` or `temperature` is out of bounds.
        """
        if questions_per_chunk < 1:
            raise ValueError("questions_per_chunk must be at least 1")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        resolved_key, resolved_url, resolved_model, resolved_ssl, resolved_conc = (
            self._resolve_connection_params(
                api_key=api_key,
                base_url=base_url,
                model=model,
                verify_ssl=verify_ssl,
                max_concurrency=max_concurrency,
            )
        )

        self.model: Final[str] = resolved_model
        self.max_concurrency: Final[int] = resolved_conc
        self.questions_per_chunk: Final[int] = questions_per_chunk
        self.temperature: Final[float] = temperature

        self._semaphore: Final[asyncio.Semaphore] = asyncio.Semaphore(resolved_conc)
        self._http_client: Final[httpx.AsyncClient] = httpx.AsyncClient(verify=resolved_ssl)
        self._client: Final[AsyncOpenAI] = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            http_client=self._http_client,
        )


    @staticmethod
    def _resolve_connection_params(
        api_key: str | None, base_url: str | None, model: str | None, 
        verify_ssl: bool | None, max_concurrency: int | None,
    ) -> tuple[str, str, str, bool, int]:
        """
        Resolves missing connection parameters lazily from application settings.

        Queries the cached `Settings` singleton only when at least one explicit
        override argument is `None`, avoiding unnecessary environment loading or
        validation errors if all connection parameters are supplied directly.

        Args:
            api_key (str | None): Explicit API authentication token, or None to
                fall back to `Settings.AI_API_KEY`.
            base_url (str | None): Explicit OpenAI-compatible endpoint URL, or
                None to fall back to `Settings.AI_BASE_URL`.
            model (str | None): Explicit target model identifier, or None to
                fall back to `Settings.AI_MODEL`.
            verify_ssl (bool | None): Explicit TLS/SSL verification flag, or None
                to fall back to `Settings.AI_VERIFY_SSL`.
            max_concurrency (int | None): Explicit maximum concurrent requests
                limit, or None to fall back to `Settings.AI_MAX_CONCURRENCY`.

        Returns:
            tuple[str, str, str, bool, int]: Resolved connection tuple ordered as
                `(api_key, base_url, model, verify_ssl, max_concurrency)`.
        """
        needs_settings = any(
            param is None
            for param in (api_key, base_url, model, verify_ssl, max_concurrency)
        )
        settings = get_settings() if needs_settings else None

        resolved_key = (
            api_key if api_key is not None else settings.AI_API_KEY.get_secret_value()  # type: ignore[union-attr]
        )

        resolved_url = (
            base_url if base_url is not None else str(settings.AI_BASE_URL)  # type: ignore[union-attr]
        )

        resolved_model = (
            model if model is not None else settings.AI_MODEL  # type: ignore[union-attr]
        )
        
        resolved_ssl = (
            verify_ssl if verify_ssl is not None else settings.AI_VERIFY_SSL  # type: ignore[union-attr]
        )

        resolved_conc = (
            max_concurrency if max_concurrency is not None else settings.AI_MAX_CONCURRENCY  # type: ignore[union-attr]
        )

        return resolved_key, resolved_url, resolved_model, resolved_ssl, resolved_conc


    async def synthesize_chunks(self, chunks: Sequence[ChunkProtocol]) -> list[SFTRecord]:
        """
        Generates SFT records concurrently across a sequence of document chunks.

        Args:
            chunks (Sequence[ChunkProtocol]): Objects exposing `chunk_id`,
                `section_title`, and `content`.

        Returns:
            list[SFTRecord]: Flattened list of validated SFT training records across
                all processed chunks.
        """
        if not chunks:
            return []

        tasks = [self.synthesize_single(chunk) for chunk in chunks]
        nested_results = await asyncio.gather(*tasks)
        
        return [record for batch in nested_results for record in batch]


    async def synthesize_single(self, chunk: ChunkProtocol) -> list[SFTRecord]:
        """
        Generates structured SFT records for a single chunk under semaphore control.

        Args:
            chunk (ChunkProtocol): Single chunk containing section title and prose.

        Returns:
            list[SFTRecord]: List of `SFTRecord` objects derived from the chunk.

        Raises:
            RuntimeError: If the LLM endpoint fails to return a valid parsed payload.
        """
        async with self._semaphore:
            parsed_response = await self._request_structured_pairs(chunk)

        return self._to_sft_records(chunk.chunk_id, parsed_response.pairs)


    async def _request_structured_pairs(
        self, chunk: ChunkProtocol
    ) -> _ChunkSynthesisResponse:
        """
        Executes the structured completion request against the OpenAI-compatible API.

        Formats system and user prompt templates with the target pair count and
        chunk context, invokes structured output parsing using
        `_ChunkSynthesisResponse`, and validates that a parsed payload was
        returned.

        Args:
            chunk (ChunkProtocol): Document chunk providing `chunk_id`,
                `section_title`, and `content` attributes for synthesis.

        Returns:
            _ChunkSynthesisResponse: Parsed Pydantic response containing the
                generated question-answer pairs.

        Raises:
            RuntimeError: If the model refuses the request or returns a response
                that fails schema parsing (`parsed is None`).
        """
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(num_pairs=self.questions_per_chunk)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            section_title=chunk.section_title,
            content=chunk.content,
        )

        completion = await self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_ChunkSynthesisResponse,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"Model returned an unparseable or refused response for chunk '{chunk.chunk_id}'"
            )

        return parsed


    @staticmethod
    def _to_sft_records(chunk_id: str, pairs: list[QAPair]) -> list[SFTRecord]:
        """
        Maps validated QA pairs into conversational SFT records.

        Transforms each `QAPair` into an `SFTRecord` containing a two-turn
        `user`/`assistant` message sequence tagged with the originating chunk ID
        and pair category.

        Args:
            chunk_id (str): Identifier of the source document chunk.
            pairs (list[QAPair]): List of validated question-answer pairs to
                convert.

        Returns:
            list[SFTRecord]: Conversational supervised fine-tuning records ready
                for serialization.
        """
        return [
            SFTRecord(
                chunk_id=chunk_id,
                category=pair.category,
                messages=[
                    ChatMessage(role="user", content=pair.instruction),
                    ChatMessage(role="assistant", content=pair.output),
                ],
            )
            for pair in pairs
        ]


    async def aclose(self) -> None:
        """
        Closes the underlying asynchronous client and HTTP transport cleanly.
        """
        await self._client.close()
        await self._http_client.aclose()


    async def __aenter__(self) -> DatasetSynthesizer:
        """
        Enters the asynchronous context manager and returns the synthesizer instance.
        """
        return self


    async def __aexit__(self, *_: object) -> None:
        """
        Exits the asynchronous context manager and closes network resources.
        """
        await self.aclose()