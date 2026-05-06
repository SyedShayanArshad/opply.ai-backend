"""
record_explanation.py — LangChain-powered email classification explainer

Generates a human-readable explanation for why an email was classified
as important or not important, shown directly to the student on the dashboard.
Uses LangChain's ChatMistralAI chain via MistralLLM wrapper.
"""
from __future__ import annotations

from .mistral_client import MistralLLM


def _fallback_explanation(*, classification: str, ex, score_text: str) -> str:
    if classification == "important":
        opportunity = ex.opportunity_type.value if ex.opportunity_type else "opportunity"
        title = ex.title or "this email"
        return (
            f"Important: {title} looks like a real {opportunity} opportunity "
            f"and aligns with your student profile. {score_text}"
        ).strip()
    return (
        "Not important: this email does not appear to be a relevant student opportunity "
        "based on your current profile and preferences."
    )


async def build_record_explanation(
    *,
    classification: str,
    profile_summary: str,
    email,
    ex,
    llm: MistralLLM,
    score=None,
    rag_context: str = "",
) -> str:
    """
    Build a concise explanation for the student via LangChain.

    rag_context: The same profile context that was used during extraction,
                 included here so the explanation references why THIS student's
                 profile matched (or didn't match) this opportunity.
    """
    score_text = (
        f"Score: {score.total:.0f}/100 "
        f"(fit {score.fit:.0f}, urgency {score.urgency:.0f}, "
        f"completeness {score.completeness:.0f})."
        if score
        else ""
    )

    if not llm.available:
        return _fallback_explanation(classification=classification, ex=ex, score_text=score_text)

    system = (
        "You write clear, student-friendly email classification explanations for Opply AI. "
        "Return strict JSON only. No markdown."
    )

    if classification == "important":
        instruction = (
            "Write 3-4 concise sentences explaining why this email is important for the student. "
            "Mention: what the opportunity is, why it matches this student's profile "
            "(reference their skills/interests/preferences from the context), "
            "and what the student should do next at a high level."
        )
    else:
        instruction = (
            "Write exactly one concise sentence explaining why this email is NOT important "
            "for this student — be specific about the mismatch (wrong type, ineligible, "
            "irrelevant to profile, etc.)."
        )

    extracted_type = (
        ex.opportunity_type.value if getattr(ex, "opportunity_type", None) else None
    )

    profile_context_section = (
        f"STUDENT PROFILE CONTEXT (RAG-retrieved):\n{rag_context}"
        if rag_context.strip()
        else f"STUDENT PROFILE SUMMARY:\n{profile_summary}"
    )

    user = (
        f"{instruction}\n\n"
        f"classification: {classification}\n"
        f"{profile_context_section}\n\n"
        f"EMAIL:\n"
        f"subject: {email.subject}\n"
        f"sender: {email.sender}\n\n"
        f"EXTRACTION RESULTS:\n"
        f"opportunity_type: {extracted_type}\n"
        f"title: {getattr(ex, 'title', None)}\n"
        f"organization: {getattr(ex, 'organization', None)}\n"
        f"summary: {getattr(ex, 'summary', None)}\n"
        f"deadline_text: {getattr(ex, 'deadline_text', None)}\n"
        f"next_steps: {getattr(ex, 'next_steps', [])}\n"
        f"eligibility: {getattr(ex, 'eligibility', [])}\n"
        f"requirements: {getattr(ex, 'requirements', [])}\n"
        f"benefits: {getattr(ex, 'benefits', [])}\n"
        f"links: {getattr(ex, 'links', [])}\n"
        f"score: {score_text}\n"
    )
    schema_hint = '{"explanation": "string"}'

    try:
        out = await llm.json_extract(system=system, user=user, schema_hint=schema_hint)
        explanation = str(out.get("explanation", "")).strip()
        if explanation:
            return explanation
    except Exception:
        pass

    return _fallback_explanation(classification=classification, ex=ex, score_text=score_text)