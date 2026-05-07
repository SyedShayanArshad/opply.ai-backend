"""
extract.py — Pure RAG-powered opportunity extraction & scoring

This module is now 100% RAG-driven. All manual heuristics, regex, and keyword
matching (rapidfuzz) have been removed.

The extraction process:
1. Receives semantically relevant profile chunks via FAISS (rag_context).
2. Uses a single LangChain/Mistral call to:
   a. Extract structured opportunity fields.
   b. Calculate a semantic 'fit_score' (0-100) based on the retrieved context.
   c. Generate detailed 'fit_reasons'.

This eliminates the 'Hybrid' model in favor of a full 'AI Reasoning' model.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .mistral_client import MistralLLM
from .models import EmailInput, OpportunityExtraction, StudentProfile
from .utils import extract_urls, parse_deadline_to_datetime, safe_str


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM = """\
You are a Senior Student Opportunity Analyst AI for Opply.

Your Goal:
1. Identify if the email contains a genuine student opportunity (scholarship, internship, competition, admissions, fellowship, or event).
2. Extract all relevant details natively using the provided tool schema.
3. Perform a rigorous SEMANTIC FIT ANALYSIS based on the provided 'RELEVANT STUDENT PROFILE CONTEXT'.

Fit Scoring Instructions (0 to 100):
- 80-100: Matches student's specific skills, interests, AND preferred opportunity types perfectly.
- 50-79: Matches some interests or skills, but might be a different opportunity type or slightly different field.
- 0-49: Low relevance or specifically excluded by their preferences.

Rules:
- If it is NOT an opportunity, set is_opportunity=false and skip other fields.
- Do NOT hallucinate deadlines or links. Only extract what is present.
"""


def _email_to_text(email: EmailInput) -> str:
    raw = safe_str(email.raw)
    if raw:
        return raw
    parts: list[str] = []
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.sender:
        parts.append(f"From: {email.sender}")
    if email.received_at:
        parts.append(f"Date: {email.received_at.isoformat()}")
    parts.append(safe_str(email.body))
    return "\n".join(p for p in parts if p)


async def extract_opportunity(
    email: EmailInput,
    profile: StudentProfile,
    *,
    base: datetime,
    llm: MistralLLM,
    notice_text: str | None = None,
    profile_summary: str | None = None,
    rag_context: str = "",
    user_id: int | None = None,
) -> OpportunityExtraction:
    """
    100% RAG-based extraction and fit analysis.
    Manual heuristics have been completely removed.
    """
    text = _email_to_text(email)

    if not llm.available:
        # If LLM is down, we return a negative extraction as we no longer support heuristic fallbacks
        return OpportunityExtraction(
            is_opportunity=False,
            extraction_warnings=["LLM Unavailable: Semantic analysis could not be performed."]
        )

    # ── Context Construction ──────────────────────────────────────────────────
    # We use RAG context if available, otherwise fall back to the profile summary
    context_to_use = rag_context.strip() or profile_summary or "No student profile context available."

    user_prompt = (
        "Analyze this email and the student context below.\n\n"
        "RELEVANT STUDENT CONTEXT:\n"
        f"{context_to_use}\n\n"
        "EMAIL TEXT:\n"
        f"{text}\n\n"
        f"OPTIONAL NOTICE TEXT:\n"
        f"{safe_str(notice_text) if notice_text else '(none)'}"
    )

    # ── LLM Extraction & Scoring ──────────────────────────────────────────────
    try:
        extraction = await llm.structured_extract(
            system=_SYSTEM, user=user_prompt, schema=OpportunityExtraction
        )

    except Exception as exc:
        import logging
        logging.getLogger("app.extract").error("Pure RAG extraction failed: %s", exc)
        return OpportunityExtraction(
            is_opportunity=False,
            extraction_warnings=[f"Analysis failed: {str(exc)}"]
        )

    # ── Semantic Post-Processing (Dates & Links only, no heuristics) ──────────
    if not extraction.links:
        extraction.links = extract_urls(text)

    deadline_text = safe_str(extraction.deadline_text)
    if deadline_text and not extraction.deadline_iso:
        dt = parse_deadline_to_datetime(deadline_text, base=base)
        if dt:
            extraction.deadline_iso = dt.isoformat()

    return extraction
