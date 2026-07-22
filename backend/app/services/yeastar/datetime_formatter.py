from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


class YeastarDateTimeFormatter:
    """Translate the supported .NET/PBX tokens without guessing their case."""

    _TOKENS = {
        "yyyy": "%Y",
        "YYYY": "%Y",
        "MM": "%m",
        "dd": "%d",
        "DD": "%d",
        "HH": "%H",
        "hh": "%I",
        "mm": "%M",
        "ss": "%S",
        "tt": "%p",
        "AM": "%p",
    }
    _ORDERED_TOKENS = tuple(sorted(_TOKENS, key=len, reverse=True))
    _LITERALS = frozenset("/-. :,T")

    @classmethod
    def dotnet_to_strftime(cls, pattern: str) -> str:
        if not pattern or not pattern.strip():
            raise ValueError("Phone-system date format is empty.")
        source = pattern.strip()
        translated: list[str] = []
        index = 0
        while index < len(source):
            token = next(
                (item for item in cls._ORDERED_TOKENS if source.startswith(item, index)),
                None,
            )
            if token is not None:
                translated.append(cls._TOKENS[token])
                index += len(token)
                continue
            character = source[index]
            if character not in cls._LITERALS:
                raise ValueError("Phone-system date format contains an unsupported token.")
            translated.append(character)
            index += 1
        result = "".join(translated)
        if "%Y" not in result or "%m" not in result or "%d" not in result:
            raise ValueError("Phone-system date format must contain year, month and day.")
        if "%H" in result and ("%I" in result or "%p" in result):
            raise ValueError("Phone-system date format mixes 12-hour and 24-hour time.")
        if "%I" in result and "%p" not in result:
            raise ValueError("A 12-hour phone-system format must include AM/PM.")
        if "%p" in result and "%I" not in result:
            raise ValueError("AM/PM requires a 12-hour phone-system format.")
        return result

    @classmethod
    def format_pattern(
        cls,
        value: datetime,
        pattern: str,
        pbx_timezone: ZoneInfo,
    ) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("CDR timestamps must include a timezone.")
        python_pattern = cls.dotnet_to_strftime(pattern)
        return value.astimezone(UTC).astimezone(pbx_timezone).strftime(python_pattern)

    def format_cdr_datetime(
        self,
        value: datetime,
        system_date_format: str,
        system_time_format: str,
        pbx_timezone: ZoneInfo,
    ) -> str:
        date_part = system_date_format.strip()
        time_part = system_time_format.strip()
        combined = f"{date_part} {time_part}".strip()
        return self.format_pattern(value, combined, pbx_timezone)


__all__ = ["YeastarDateTimeFormatter"]
