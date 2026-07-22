from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator
from typing import Any


FORMULA_PREFIXES = ("=", "+", "-", "@")


def csv_safe_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    stripped = text.lstrip(" \t\r\n")
    if stripped.startswith(FORMULA_PREFIXES):
        return "'" + text
    return text


def csv_bytes(rows: Iterable[Iterable[Any]]) -> Iterator[bytes]:
    yield b"\xef\xbb\xbf"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, dialect="excel", lineterminator="\r\n")
    for row in rows:
        writer.writerow([csv_safe_cell(value) for value in row])
        yield buffer.getvalue().encode("utf-8")
        buffer.seek(0)
        buffer.truncate(0)


def mask_phone_number(value: str | None) -> str | None:
    if not value:
        return None
    characters = list(value)
    digit_positions = [index for index, character in enumerate(characters) if character.isdigit()]
    visible = set(digit_positions[-4:])
    return "".join(
        character if index in visible or not character.isdigit() else "•"
        for index, character in enumerate(characters)
    )
