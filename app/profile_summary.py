from __future__ import annotations

from .mistral_client import MistralLLM
from .models import StudentProfile
from .utils import safe_str


def fallback_profile_summary(profile: StudentProfile) -> str:
    skills = ", ".join(profile.skills[:6]) if profile.skills else "not specified"
    interests = ", ".join(profile.interests[:6]) if profile.interests else "not specified"
    preferred = ", ".join([p.value for p in profile.preferred_opportunity_types[:6]]) if profile.preferred_opportunity_types else "not specified"
    location = profile.location_text or profile.location_preference.value
    exp = profile.past_experience or "not specified"
    return (
        f"Student is in {profile.degree_program}, semester {profile.semester}, with CGPA {profile.cgpa:.2f}. "
        f"Skills: {skills}. Interests: {interests}. Preferred opportunity types: {preferred}. "
        f"Financial need: {profile.financial_need.value}. Location preference: {location}. "
        f"Past experience: {exp}."
    )


async def build_profile_summary(profile: StudentProfile, llm: MistralLLM) -> str:
    if not llm.available:
        return fallback_profile_summary(profile)

    system = (
        "You are a profile writer for student opportunity matching. "
        "Return strict JSON only. Do not add markdown."
    )
    user = (
        "Create a concise but detailed student profile summary (90-140 words) from this student data. "
        "Write in third person, neutral professional tone. Focus on study background, strengths, goals, "
        "preferred opportunities, readiness signals, and anything that helps classify opportunities.\n\n"
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

    return fallback_profile_summary(profile)