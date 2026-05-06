"""
profile_summary.py — LangChain-powered student profile summariser

Generates a concise professional summary of the student profile that is
stored in DB and reused across all email processing cycles.
Uses LangChain's ChatPromptTemplate chain via MistralLLM wrapper.
"""
from __future__ import annotations

from .mistral_client import MistralLLM
from .models import StudentProfile


def _fallback_summary(profile: StudentProfile) -> str:
    """Plain-text summary used when LLM is unavailable."""
    skills = ", ".join(profile.skills[:6]) if profile.skills else "not specified"
    interests = ", ".join(profile.interests[:6]) if profile.interests else "not specified"
    preferred = (
        ", ".join(p.value for p in profile.preferred_opportunity_types[:6])
        if profile.preferred_opportunity_types
        else "not specified"
    )
    loc = profile.location_text or profile.location_preference.value
    exp = profile.past_experience or "not specified"
    return (
        f"Student is in {profile.degree_program}, semester {profile.semester}, "
        f"CGPA {profile.cgpa:.2f}. Skills: {skills}. Interests: {interests}. "
        f"Preferred opportunity types: {preferred}. "
        f"Financial need: {profile.financial_need.value}. "
        f"Location preference: {loc}. Past experience: {exp}."
    )


async def build_profile_summary(profile: StudentProfile, llm: MistralLLM) -> str:
    """
    Generate a 90-140 word professional student profile summary via LangChain.
    Falls back to plain-text summary if LLM is unavailable.
    """
    if not llm.available:
        return _fallback_summary(profile)

    system = (
        "You are a professional profile writer for a student opportunity-matching platform. "
        "Return strict JSON only. Do not add markdown or extra commentary."
    )
    user = (
        "Write a concise but detailed student profile summary (90-140 words) from the data below.\n"
        "Write in third person, neutral professional tone.\n"
        "Focus on: study background, key strengths, career goals, preferred opportunities, "
        "financial situation, and anything that helps classify relevant emails.\n\n"
        f"degree_program: {profile.degree_program}\n"
        f"semester: {profile.semester}\n"
        f"cgpa: {profile.cgpa}\n"
        f"skills: {profile.skills}\n"
        f"interests: {profile.interests}\n"
        f"preferred_opportunity_types: {[p.value for p in profile.preferred_opportunity_types]}\n"
        f"financial_need: {profile.financial_need.value}\n"
        f"location_preference: {profile.location_preference.value}\n"
        f"location_text: {profile.location_text}\n"
        f"past_experience: {profile.past_experience}\n"
    )
    schema_hint = '{"profile_summary": "string"}'

    try:
        out = await llm.json_extract(system=system, user=user, schema_hint=schema_hint)
        summary = str(out.get("profile_summary", "")).strip()
        if summary:
            return summary
    except Exception:
        pass

    return _fallback_summary(profile)