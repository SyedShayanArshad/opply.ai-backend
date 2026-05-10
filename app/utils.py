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


import urllib.parse
from datetime import timedelta

def generate_google_calendar_url(title: str, organization: str, summary: str, location: str, deadline_iso: str | None, links: list[str]) -> str:
    if not deadline_iso:
        return ""
    try:
        # Expected format: ISO-8601 string, e.g. "2026-06-15T00:00:00Z"
        # Google calendar all-day event format: YYYYMMDD/YYYYMMDD
        dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
        start_date = dt.strftime("%Y%m%d")
        end_date = (dt + timedelta(days=1)).strftime("%Y%m%d")
        
        parts = []
        if organization: parts.append(f"Organization: {organization}")
        if summary: parts.append(f"\nSummary:\n{summary}")
        if links: parts.append(f"\nLinks:\n" + "\n".join(links[:3]))
        parts.append("\n— Added via Opply AI")
        
        params = {
            "action": "TEMPLATE",
            "text": f"📌 {title}",
            "dates": f"{start_date}/{end_date}",
            "details": "\n".join(parts),
            "location": location or ""
        }
        return "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode(params)
    except Exception as e:
        return ""
