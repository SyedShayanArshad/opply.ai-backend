from __future__ import annotations

import re
from datetime import datetime, timezone

import dateparser


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_deadline_to_datetime(deadline_text: str | None, *, base: datetime) -> datetime | None:
    if not deadline_text:
        return None

    dt = dateparser.parse(
        deadline_text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": base,
        },
    )
    if not dt:
        return None
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_URL_RE = re.compile(r"https?://[^\s)\]}>,\"']+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return list(dict.fromkeys(_URL_RE.findall(text)))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_str(x: str | None) -> str:
    return (x or "").strip()
