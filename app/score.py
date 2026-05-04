from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rapidfuzz import fuzz

from .models import OpportunityExtraction, OpportunityType, ScoreBreakdown, StudentProfile
from .utils import clamp, parse_deadline_to_datetime, safe_str


def _token_set(text: str) -> set[str]:
    return {t for t in safe_str(text).lower().replace("/", " ").split() if len(t) >= 2}


def _match_keywords(haystack: str, needles: list[str]) -> float:
    if not haystack or not needles:
        return 0.0
    h = haystack.lower()
    best = 0.0
    for n in needles:
        n = n.strip()
        if not n:
            continue
        best = max(best, float(fuzz.partial_ratio(h, n.lower())))
    return best / 100.0


def _urgency_score(deadline_dt: datetime | None, *, base: datetime) -> tuple[float, int | None, bool]:
    if not deadline_dt:
        return (25.0, None, False)

    delta = deadline_dt - base
    days = int(delta.total_seconds() // 86400)
    if delta.total_seconds() < 0:
        return (0.0, days, True)

    if days <= 3:
        return (100.0, days, False)
    if days <= 7:
        return (90.0, days, False)
    if days <= 14:
        return (75.0, days, False)
    if days <= 30:
        return (50.0, days, False)
    return (25.0, days, False)


def _completeness(ex: OpportunityExtraction) -> float:
    fields = [
        bool(ex.title),
        bool(ex.opportunity_type),
        bool(ex.deadline_text or ex.deadline_iso),
        bool(ex.links or ex.contact),
        bool(ex.eligibility),
        bool(ex.required_documents or ex.requirements),
        bool(ex.next_steps),
    ]
    return 100.0 * (sum(1 for f in fields if f) / len(fields))


def _fit_score(profile: StudentProfile, ex: OpportunityExtraction) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 50.0  # neutral baseline

    # Preferred type
    if profile.preferred_opportunity_types and ex.opportunity_type:
        if ex.opportunity_type in profile.preferred_opportunity_types:
            score += 20
            reasons.append("Matches preferred opportunity type")
        else:
            score -= 10
            reasons.append("Not in preferred opportunity types")

    # Financial need boost for scholarships/funding
    if ex.opportunity_type in {OpportunityType.scholarship, OpportunityType.fellowship}:
        if profile.financial_need.value == "high":
            score += 15
            reasons.append("Financial need aligns with funded opportunity")
        elif profile.financial_need.value == "low":
            score -= 2

    # Skills + interests textual match
    blob = " ".join(
        [
            safe_str(ex.title),
            safe_str(ex.organization),
            " ".join(ex.requirements),
            " ".join(ex.eligibility),
            " ".join(ex.benefits),
        ]
    )

    skill_match = _match_keywords(blob, profile.skills)
    interest_match = _match_keywords(blob, profile.interests)

    if skill_match >= 0.8:
        score += 15
        reasons.append("Strong match to your skills")
    elif skill_match >= 0.5:
        score += 8
        reasons.append("Some match to your skills")

    if interest_match >= 0.8:
        score += 10
        reasons.append("Strong match to your interests")
    elif interest_match >= 0.5:
        score += 5
        reasons.append("Some match to your interests")

    # CGPA eligibility: very lightweight parsing for common patterns
    elig_text = " ".join(ex.eligibility).lower()
    min_cgpa = None
    for pattern in ["cgpa", "gpa"]:
        if pattern in elig_text:
            import re

            m = re.search(r"(?:cgpa|gpa)\s*(?:>=|>|at\s*least|minimum)?\s*(\d+(?:\.\d+)?)", elig_text)
            if m:
                try:
                    min_cgpa = float(m.group(1))
                except Exception:
                    min_cgpa = None
            break

    if min_cgpa is not None:
        if profile.cgpa + 1e-6 >= min_cgpa:
            score += 8
            reasons.append(f"Meets minimum CGPA ({min_cgpa})")
        else:
            score -= 25
            reasons.append(f"CGPA may be below minimum ({min_cgpa})")

    # Semester signals
    if ex.opportunity_type == OpportunityType.internship:
        if profile.semester <= 2:
            score -= 5
            reasons.append("Early semester may limit internship eligibility")
        elif profile.semester >= 5:
            score += 5
            reasons.append("Semester level fits typical internships")

    return (clamp(score, 0.0, 100.0), reasons)


def score_opportunity(
    profile: StudentProfile,
    ex: OpportunityExtraction,
    *,
    base: datetime,
) -> tuple[ScoreBreakdown, list[str]]:
    deadline_dt = None
    if ex.deadline_iso:
        try:
            deadline_dt = datetime.fromisoformat(ex.deadline_iso)
        except Exception:
            deadline_dt = None
    if not deadline_dt and ex.deadline_text:
        deadline_dt = parse_deadline_to_datetime(ex.deadline_text, base=base)

    fit, fit_reasons = _fit_score(profile, ex)
    urgency, days, expired = _urgency_score(deadline_dt, base=base)
    completeness = _completeness(ex)

    total = clamp(0.50 * fit + 0.35 * urgency + 0.15 * completeness, 0.0, 100.0)

    reasons = []
    reasons.extend(fit_reasons[:3])
    if days is not None:
        if expired:
            reasons.append("Deadline appears to have passed")
        else:
            reasons.append(f"Deadline in ~{max(days, 0)} day(s)")
    else:
        reasons.append("No clear deadline found (lower urgency confidence)")

    score = ScoreBreakdown(
        fit=fit,
        urgency=urgency,
        completeness=completeness,
        total=total,
        days_to_deadline=days,
        expired=expired,
    )

    return score, reasons
