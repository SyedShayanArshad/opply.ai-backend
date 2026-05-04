from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from .mistral_client import MistralLLM
from .models import EmailInput, OpportunityExtraction, OpportunityType, StudentProfile
from .utils import extract_urls, parse_deadline_to_datetime, safe_str


_SYSTEM = """You are an information extraction engine for university opportunity emails.
You must:
- Decide if the email contains a real opportunity (scholarship/internship/competition/admissions/fellowship/event).
- If yes, extract structured fields.
- Provide a detailed summary of what the email is about in the 'summary' field.
- Provide detailed, step-by-step required actions in the 'next_steps' field.
- Provide evidence quotes per field (short exact snippets).
- If a field is missing, set it to null or empty.
Return strict JSON only.
"""

_SCHEMA_HINT = """{
  "is_opportunity": true,
  "opportunity_type": "scholarship|internship|competition|admissions|fellowship|event|other|null",
  "title": "string|null",
  "organization": "string|null",
  "summary": "string|null",
  "deadline_text": "string|null",
  "location": "string|null",
  "eligibility": ["string"],
  "required_documents": ["string"],
  "links": ["https://..."],
  "contact": "string|null",
  "next_steps": ["string"],
  "requirements": ["string"],
  "benefits": ["string"],
  "evidence": {"field": ["quote"]}
}"""


def _email_to_text(email: EmailInput) -> str:
    raw = safe_str(email.raw)
    if raw:
        return raw
    parts = []
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.sender:
        parts.append(f"From: {email.sender}")
    if email.received_at:
        parts.append(f"Date: {email.received_at.isoformat()}")
    parts.append(safe_str(email.body))
    return "\n".join([p for p in parts if p])


def _heuristic_is_opportunity(text: str) -> bool:
    keywords = [
        "scholarship",
        "internship",
        "apply",
        "application",
        "deadline",
        "fellowship",
        "competition",
        "call for",
        "admission",
        "fully funded",
        "stipend",
    ]
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    if hits >= 2:
        return True
    if "unsubscribe" in t and hits == 0:
        return False
    return hits >= 1 and len(text) > 200


def _guess_type(text: str) -> OpportunityType:
    t = text.lower()
    if "scholarship" in t:
        return OpportunityType.scholarship
    if "internship" in t:
        return OpportunityType.internship
    if "competition" in t or "challenge" in t or "hackathon" in t:
        return OpportunityType.competition
    if "admission" in t or "admissions" in t:
        return OpportunityType.admissions
    if "fellowship" in t:
        return OpportunityType.fellowship
    if "workshop" in t or "webinar" in t or "event" in t:
        return OpportunityType.event
    return OpportunityType.other


_DEADLINE_RE = re.compile(
    r"(?:deadline|last date|apply by)\s*[:\-]?\s*(.+)", re.IGNORECASE
)


def _heuristic_extract(email: EmailInput, *, base: datetime) -> OpportunityExtraction:
    text = _email_to_text(email)
    urls = extract_urls(text)

    deadline_text = None
    m = _DEADLINE_RE.search(text)
    if m:
        deadline_text = m.group(1).strip()[:120]

    op_type = _guess_type(text)
    is_opp = _heuristic_is_opportunity(text)

    extraction = OpportunityExtraction(
        is_opportunity=is_opp,
        opportunity_type=op_type if is_opp else None,
        title=email.subject or None,
        organization=None,
        summary="A generic opportunity extracted heuristically (LLM unavailable).",
        deadline_text=deadline_text,
        location=None,
        eligibility=[],
        required_documents=[],
        links=urls,
        contact=None,
        next_steps=["Open the link and confirm eligibility", "Prepare required documents"],
        requirements=[],
        benefits=[],
        evidence={},
        extraction_warnings=["Heuristic extraction used (no Mistral API key configured)"]
        if not is_opp
        else ["Heuristic extraction used (no Mistral API key configured)"]
    )

    if deadline_text:
        dt = parse_deadline_to_datetime(deadline_text, base=base)
        if dt:
            extraction.deadline_iso = dt.isoformat()

    return extraction


async def extract_opportunity(
    email: EmailInput,
    profile: StudentProfile,
    *,
    base: datetime,
    llm: MistralLLM,
    notice_text: str | None = None,
    profile_summary: str | None = None,
) -> OpportunityExtraction:
    text = _email_to_text(email)

    if not llm.available:
        return _heuristic_extract(email, base=base)

    user = (
        "Extract opportunity fields from this email. "
        "If it is NOT an opportunity, set is_opportunity=false and keep other fields null/empty.\n\n"
        f"STUDENT CONTEXT (for disambiguation only; do not invent):\n"
        f"- degree_program: {profile.degree_program}\n"
        f"- semester: {profile.semester}\n"
        f"- cgpa: {profile.cgpa}\n"
        f"- profile_summary: {safe_str(profile_summary) if profile_summary else ''}\n\n"
        f"EMAIL TEXT:\n{text}\n\n"
        f"OPTIONAL NOTICE TEXT (if relevant):\n{safe_str(notice_text) if notice_text else ''}"
    )

    data: dict[str, Any] = {}
    try:
        data = await llm.json_extract(system=_SYSTEM, user=user, schema_hint=_SCHEMA_HINT)
    except Exception:
        # fall back
        return _heuristic_extract(email, base=base)

    # Coerce/clean
    try:
        extraction = OpportunityExtraction.model_validate(data)
    except Exception:
        return _heuristic_extract(email, base=base)

    # Enrich: URLs if missing
    if not extraction.links:
        extraction.links = extract_urls(text)

    # Normalize type
    if extraction.is_opportunity and not extraction.opportunity_type:
        extraction.opportunity_type = _guess_type(text)

    # Parse deadline
    deadline_text = safe_str(extraction.deadline_text)
    if deadline_text and not extraction.deadline_iso:
        dt = parse_deadline_to_datetime(deadline_text, base=base)
        if dt:
            extraction.deadline_iso = dt.isoformat()

    # If model says not opportunity but subject strongly matches, override conservatively
    subj = safe_str(email.subject).lower()
    if not extraction.is_opportunity and subj:
        if max(
            fuzz.partial_ratio(subj, "scholarship"),
            fuzz.partial_ratio(subj, "internship"),
            fuzz.partial_ratio(subj, "fellowship"),
        ) > 85:
            extraction.is_opportunity = True
            extraction.extraction_warnings.append("Overrode is_opportunity via subject keyword")
            if not extraction.opportunity_type:
                extraction.opportunity_type = _guess_type(text)

    return extraction
