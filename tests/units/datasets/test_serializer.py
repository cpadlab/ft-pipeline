
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ft.datasets.serializer import JSONLSerializer
from ft.datasets.synthesizer import ChatMessage, SFTRecord


@pytest.mark.parametrize("bad_name", ["", "dataset.json", "sub/dataset.jsonl"])
def test_serializer_invalid_default_filename(bad_name: str) -> None:
    with pytest.raises(ValueError, match="default_filename"):
        JSONLSerializer(default_filename=bad_name)


def test_save_deduplicates_and_includes_metadata(
    tmp_path: Path, sample_sft_records: list[SFTRecord]
) -> None:
    dup = SFTRecord(
        chunk_id="doc_p003",
        category="architecture",
        messages=[
            ChatMessage(role="user", content="  explain   the CORE architecture. "),
            ChatMessage(role="assistant", content="Duplicate answer."),
        ],
    )
    short_msg = SimpleNamespace(
        chunk_id="doc_p004",
        category="debugging",
        messages=[ChatMessage(role="user", content="Only one turn")],
    )
    blank_turn = SFTRecord(
        chunk_id="doc_p005",
        category="debugging",
        messages=[
            ChatMessage(role="user", content="   "),
            ChatMessage(role="assistant", content="Answer"),
        ],
    )

    serializer = JSONLSerializer(include_metadata=True, deduplicate=True)
    out = serializer.save(
        [*sample_sft_records, dup, short_msg, blank_turn],
        tmp_path / "nested_out",
        create_dirs=True,
    )

    assert out == tmp_path / "nested_out" / "dataset.jsonl"
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["chunk_id"] == "doc_p001"
    assert lines[0]["category"] == "architecture"


def test_save_append_and_overwrite_and_no_deduplicate(
    tmp_path: Path, sample_sft_records: list[SFTRecord]
) -> None:
    target = tmp_path / "out.jsonl"
    serializer = JSONLSerializer(deduplicate=True)
    serializer.save([sample_sft_records[0]], target)

    with target.open("a", encoding="utf-8") as fh:
        fh.write("\nnot-json\n{}\n")

    serializer.save(sample_sft_records, target, append=True)
    assert len(target.read_text(encoding="utf-8").splitlines()) == 5

    no_dedup = JSONLSerializer(deduplicate=False)
    no_dedup.save([sample_sft_records[0], sample_sft_records[0]], target, overwrite=True)
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2
    assert no_dedup._load_existing_prompts(target) == set()


def test_save_error_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_sft_records: list[SFTRecord]
) -> None:
    serializer = JSONLSerializer()
    out = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="cannot both be True"):
        serializer.save(sample_sft_records, out, overwrite=True, append=True)

    with pytest.raises(ValueError, match="No valid records"):
        serializer.save([], out)
    assert not out.exists()

    out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        serializer.save(sample_sft_records, out)

    out_dir = tmp_path / "out_dir"
    (out_dir / "dataset.jsonl").mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        serializer.save(sample_sft_records, out_dir)

    plain_file = tmp_path / "plain"
    plain_file.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        serializer.save(sample_sft_records, plain_file)

    with pytest.raises(NotADirectoryError):
        serializer.save(sample_sft_records, plain_file / "sub.jsonl")

    with pytest.raises(FileNotFoundError):
        serializer.save(sample_sft_records, tmp_path / "missing_dir" / "a.jsonl", create_dirs=False)

    monkeypatch.setattr(os, "access", lambda *_args: False)
    with pytest.raises(PermissionError):
        serializer.save(sample_sft_records, tmp_path / "perm.jsonl")


def test_register_prompt_empty() -> None:
    seen: set[str] = set()
    assert JSONLSerializer._register_prompt("   ", seen) is False
