from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalize_greek(text: str, *, remove_accents: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ς", "σ")
    if remove_accents:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
        text = unicodedata.normalize("NFC", text)
    characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith(("P", "S", "Z", "C")):
            characters.append(" ")
        else:
            characters.append(character)
    return _WHITESPACE.sub(" ", "".join(characters)).strip()


def normalize_greek_with_map(
    text: str, *, remove_accents: bool = True
) -> tuple[str, list[int]]:
    """Normalize text and retain a source-character index for every output character."""
    raw_characters: list[str] = []
    raw_map: list[int] = []
    for source_index, original_character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", original_character).casefold().replace("ς", "σ")
        if remove_accents:
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFD", normalized)
                if unicodedata.category(character) != "Mn"
            )
        for character in normalized:
            category = unicodedata.category(character)
            raw_characters.append(" " if category.startswith(("P", "S", "Z", "C")) else character)
            raw_map.append(source_index)
    result: list[str] = []
    index_map: list[int] = []
    pending_space_index: int | None = None
    for character, source_index in zip(raw_characters, raw_map, strict=True):
        if character.isspace():
            if result:
                pending_space_index = source_index
            continue
        if pending_space_index is not None:
            result.append(" ")
            index_map.append(pending_space_index)
            pending_space_index = None
        result.append(character)
        index_map.append(source_index)
    return "".join(result), index_map
