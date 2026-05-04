from __future__ import annotations

from .mistral_client import MistralLLM


def _fallback_record_explanation(*, classification: str, ex, score_text: str) -> str:
    if classification == "important":
        opportunity = ex.opportunity_type.value if ex.opportunity_type else "opportunity"
        title = ex.title or "this email"
        return (
            f"Important: {title} looks like a real {opportunity} opportunity and aligns with your student profile. "
            f"{score_text}".strip()
        )

    return (
        "Not important: this email does not read like a real student opportunity and appears unrelated to your profile goals."
    )


async def build_record_explanation(
    *,
    classification: str,
    profile_summary: str,
    email,
    ex,
    llm: MistralLLM,
    score=None,
) -> str:
    score_text = (
        f"Score: {score.total:.0f}/100. Fit {score.fit:.0f}, urgency {score.urgency:.0f}, completeness {score.completeness:.0f}."
        if score
        else ""
    )

    if not llm.available:
        return _fallback_record_explanation(classification=classification, ex=ex, score_text=score_text)

    system = (
        "You explain email classification results for Opply AI. "
        "Return strict JSON only. Do not add markdown."
    )

    if classification == "important":
        instruction = (
            "Write 3-4 concise sentences for why this email is important for the student. "
            "Mention what the opportunity is, why it was selected for this profile, and what the student should do next at a high level."
        )
    else:
        instruction = (
            "Write only one concise sentence explaining why this email is not important for this student profile."
        )

    extracted_type = ex.opportunity_type.value if getattr(ex, "opportunity_type", None) else None
    user = (
        f"{instruction}\n\n"
        f"classification: {classification}\n"
        f"student_profile_summary: {profile_summary}\n"
        f"subject: {email.subject}\n"
        f"sender: {email.sender}\n"
        f"opportunity_type: {extracted_type}\n"
        f"title: {getattr(ex, 'title', None)}\n"
        f"organization: {getattr(ex, 'organization', None)}\n"
        f"summary: {getattr(ex, 'summary', None)}\n"
        f"deadline_text: {getattr(ex, 'deadline_text', None)}\n"
        f"next_steps: {getattr(ex, 'next_steps', [])}\n"
        f"links: {getattr(ex, 'links', [])}\n"
        f"eligibility: {getattr(ex, 'eligibility', [])}\n"
        f"requirements: {getattr(ex, 'requirements', [])}\n"
        f"benefits: {getattr(ex, 'benefits', [])}\n"
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

    return _fallback_record_explanation(classification=classification, ex=ex, score_text=score_text)