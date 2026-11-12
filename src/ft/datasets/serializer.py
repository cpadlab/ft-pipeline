
from __future__ import annotations
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable


DEFAULT_OUTPUT_FILENAME: Final[str] = "dataset.jsonl"


@runtime_checkable
class MessageProtocol(Protocol):
    """
    Structural contract for a conversational turn.
    """

    @property
    def role(self) -> Literal["user", "assistant"]: ...

    @property
    def content(self) -> str: ...


@runtime_checkable
class SFTRecordProtocol(Protocol):
    """
    Structural contract decoupling serialization from the synthesizer module.
    """

    @property
    def chunk_id(self) -> str: ...

    @property
    def category(self) -> str: ...

    @property
    def messages(self) -> Sequence[MessageProtocol]: ...


class JSONLSerializer:
    """
    Validates, deduplicates, and persists SFT training records in JSONL format.

    Enforces filesystem safety checks (preventing silent overwrites, validating
    directory write permissions, and avoiding empty output artifacts) while
    producing line-delimited JSON files ready for Hugging Face `datasets`.

    Attributes:
        include_metadata (Final[bool]): Whether to retain `chunk_id` and `category`
            alongside `messages` in exported records.
        deduplicate (Final[bool]): Whether to filter records with identical user prompts.
        default_filename (Final[str]): Fallback filename when `save()` receives a
            directory path.
    """

    def __init__(
        self,
        include_metadata: bool = False,
        deduplicate: bool = True,
        default_filename: str = DEFAULT_OUTPUT_FILENAME,
    ) -> None:
        """
        Initializes the JSONLSerializer.

        Args:
            include_metadata (bool, optional): If True, includes `chunk_id` and
                `category` in each JSONL line. Defaults to False.
            deduplicate (bool, optional): If True, skips duplicate records based
                on normalized user prompt text. Defaults to True.
            default_filename (str, optional): Filename used when a directory path
                is passed to `save()`. Must end with `.jsonl`. Defaults to
                `"dataset.jsonl"`.

        Raises:
            ValueError: If `default_filename` is empty, contains path separators,
                or does not end with `.jsonl`.
        """
        if (
            not default_filename
            or not default_filename.endswith(".jsonl")
            or Path(default_filename).name != default_filename
        ):
            raise ValueError(
                "default_filename must be a valid filename (not a path) ending with '.jsonl'"
            )

        self.include_metadata: Final[bool] = include_metadata
        self.deduplicate: Final[bool] = deduplicate
        self.default_filename: Final[str] = default_filename

    
    def save(
        self, records: Iterable[SFTRecordProtocol], output_path: str | Path,
        *, overwrite: bool = False, append: bool = False, create_dirs: bool = True,
    ) -> Path:
        """
        Validates target filesystem paths and serializes SFT records to `.jsonl`.

        Args:
            records (Iterable[SFTRecordProtocol]): Sequence or stream of SFT records.
            output_path (str | Path): Destination `.jsonl` file path or target
                directory path.
            overwrite (bool, optional): If True, allows replacing an existing target
                file when `append=False`. Defaults to False.
            append (bool, optional): If True, appends new records to an existing
                file instead of overwriting. Mutually exclusive with `overwrite`.
                Defaults to False.
            create_dirs (bool, optional): If True, creates missing parent directories
                automatically. If False, raises `FileNotFoundError` when the target
                directory does not exist. Defaults to True.

        Returns:
            Path: Resolved path to the written `.jsonl` file.

        Raises:
            ValueError: If both `overwrite` and `append` are True, or if no valid
                records remain after filtering.
            FileExistsError: If the target file already exists and neither
                `overwrite` nor `append` is enabled, or if a regular file blocks
                directory creation.
            FileNotFoundError: If `create_dirs=False` and the parent directory
                does not exist.
            IsADirectoryError: If the resolved `.jsonl` target path is an existing
                directory.
            PermissionError: If the target directory is not writable.
        """
        if overwrite and append:
            raise ValueError("'overwrite' and 'append' cannot both be True")

        target = self._prepare_target_path(
            Path(output_path),
            overwrite=overwrite,
            append=append,
            create_dirs=create_dirs,
        )

        seen_prompts: set[str] = self._load_existing_prompts(target) if append else set()
        mode = "a" if append else "w"
        existed_before = target.exists()
        written_count = 0

        with target.open(mode, encoding="utf-8") as file_handle:
            for record in records:
                payload = self._serialize_record(record, seen_prompts)
                
                if payload is not None:
                    file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written_count += 1

        if written_count == 0 and not existed_before:
            target.unlink(missing_ok=True)
            raise ValueError("No valid records were serialized; output file was not created")

        return target


    def _prepare_target_path(
        self, raw_path: Path, *, overwrite: bool, append: bool, create_dirs: bool,
    ) -> Path:
        """
        Resolves file vs. directory paths and enforces filesystem preconditions.

        Normalizes `raw_path` into a concrete `.jsonl` file target, verifies or
        creates parent directories, and guards against unintended overwrites or
        directory collisions.

        Args:
            raw_path (Path): Unnormalized target file or directory path.
            overwrite (bool): Whether replacing an existing target file is allowed.
            append (bool): Whether appending to an existing target file is allowed.
            create_dirs (bool): Whether missing parent directories should be created.

        Returns:
            Path: Validated target `.jsonl` file path ready for opening.

        Raises:
            IsADirectoryError: If the resolved file path points to an existing
                directory.
            FileExistsError: If the target file already exists and both `overwrite`
                and `append` are False.
        """
        target = self._resolve_target_file(raw_path)
        self._ensure_parent_directory(target.parent, create_dirs=create_dirs)

        if target.exists():
            if target.is_dir():
                raise IsADirectoryError(f"Target path is a directory, expected a file: {target}")

            if not overwrite and not append:
                raise FileExistsError(f"Output file already exists: {target}. Pass overwrite=True or append=True.")

        return target


    def _resolve_target_file(self, path: Path) -> Path:
        """
        Maps directory paths to `<dir>/<default_filename>` while detecting path conflicts.

        Args:
            path (Path): Input path pointing either to a target `.jsonl` file or
                an output directory.

        Returns:
            Path: Concrete `.jsonl` file path.

        Raises:
            FileExistsError: If `path` exists as a regular file whose extension is
                not `.jsonl`, preventing it from being used as a directory.
        """
        if path.exists():
            if path.is_dir():
                return path / self.default_filename
            
            if path.suffix.lower() != ".jsonl":
                raise FileExistsError(f"Cannot treat '{path}' as an output directory because a file with that name exists")

            return path

        return path if path.suffix.lower() == ".jsonl" else path / self.default_filename


    @staticmethod
    def _ensure_parent_directory(parent: Path, *, create_dirs: bool) -> None:
        """Validates or creates the parent directory and checks write permissions.

        Args:
            parent (Path): Directory path that will contain the output file.
            create_dirs (bool): If True, creates missing directories recursively.

        Raises:
            NotADirectoryError: If `parent` exists on disk as a non-directory file.
            FileNotFoundError: If `parent` does not exist and `create_dirs` is False.
            PermissionError: If the current process lacks write access (`os.W_OK`)
                to `parent`.
        """
        if parent.exists():
            if not parent.is_dir():
                raise NotADirectoryError(f"Parent path is not a directory: {parent}")
        elif create_dirs:
            parent.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Target directory does not exist: {parent}")

        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Target directory is not writable: {parent}")


    def _serialize_record(
        self,
        record: SFTRecordProtocol,
        seen_prompts: set[str],
    ) -> dict[str, Any] | None:
        """
        Validates non-empty message turns, checks duplicates, and builds the dict payload.

        Ensures the record contains at least a user prompt and assistant response
        with non-whitespace content, optionally filters duplicate user prompts
        against `seen_prompts`, and attaches metadata fields if configured.

        Args:
            record (SFTRecordProtocol): Input SFT record to validate and transform.
            seen_prompts (set[str]): Mutable set of normalized user prompts already
                processed, updated in place when deduplication is active.

        Returns:
            dict[str, Any] | None: JSON-serializable dictionary payload, or None if
                the record is invalid or identified as a duplicate.
        """
        if len(record.messages) < 2:
            return None

        user_prompt = record.messages[0].content.strip()
        assistant_reply = record.messages[1].content.strip()
        if not user_prompt or not assistant_reply:
            return None

        if self.deduplicate and not self._register_prompt(user_prompt, seen_prompts):
            return None

        payload: dict[str, Any] = {
            "messages": [
                {"role": msg.role, "content": msg.content.strip()}
                for msg in record.messages
            ]
        }

        if self.include_metadata:
            payload["chunk_id"] = record.chunk_id
            payload["category"] = record.category

        return payload


    @staticmethod
    def _register_prompt(prompt: str, seen_prompts: set[str]) -> bool:
        """Normalizes a prompt string and registers it if unseen.

        Applies lowercase conversion and internal whitespace collapsing before
        checking membership in `seen_prompts`.

        Args:
            prompt (str): Raw user prompt text.
            seen_prompts (set[str]): Mutable set tracking normalized prompts seen
                so far.

        Returns:
            bool: True if the normalized prompt is non-empty and newly added;
                False if empty or already present.
        """
        normalized = " ".join(prompt.lower().split())
        if not normalized or normalized in seen_prompts:
            return False
        
        seen_prompts.add(normalized)
        
        return True


    def _load_existing_prompts(self, target: Path) -> set[str]:
        """
        Reads existing user prompts when appending to avoid cross-run duplicates.

        Args:
            target (Path): Path to an existing `.jsonl` file being appended to.

        Returns:
            set[str]: Set of normalized user prompt strings extracted from valid
                lines in `target`. Returns an empty set if deduplication is
                disabled or the file does not exist.
        """
        if not self.deduplicate or not target.exists():
            return set()

        existing: set[str] = set()
        with target.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                prompt = self._extract_prompt_from_json_line(line)
                if prompt:
                    existing.add(prompt)
        
        return existing


    @staticmethod
    def _extract_prompt_from_json_line(line: str) -> str:
        """
        Parses a single JSONL line and extracts its normalized user prompt.

        Safely ignores blank lines, malformed JSON entries, or records missing
        a valid first message dictionary.

        Args:
            line (str): Raw single line read from a `.jsonl` file.

        Returns:
            str: Lowercased, whitespace-normalized content of the first message,
                or an empty string if extraction fails.
        """
        stripped = line.strip()
        
        if not stripped:
            return ""
        
        
        try:
            data = json.loads(stripped)
            messages = data.get("messages", [])
        
            if messages and isinstance(messages[0], dict):
                raw_content = str(messages[0].get("content", ""))
        
                return " ".join(raw_content.lower().split())
        
        except json.JSONDecodeError:
            return ""
        
        return ""