"""
score.py — Semantic Scoring with RAG

This module has been stripped of all manual heuristics (regex, keyword matching,
hardcoded GPA/semester rules).

It now performs a simple weighted calculation:
1. Fit (50%): Provided directly by the LLM's semantic analysis in extract.py.
2. Urgency (35%): Calculated mathematically from the extracted deadline.
3. Completeness (15%): Calculated based on the number of non-null extracted fields.

The 'fit' component is now 100% semantic and RAG-driven.
"""
from __future__ import annotations

from datetime import datetime, timezone
from .models import OpportunityExtraction, ScoreBreakdown, StudentProfile
from .utils import clamp, parse_deadline_to_datetime


def _urgency_score(
    deadline_dt: datetime | None, *, base: datetime
) -> tuple[float, int | None, bool]:
    """Mathematical urgency calculation (not a heuristic)."""
    if not deadline_dt:
        return (25.0, None, False)
    
    delta = deadline_dt - base
    days = int(delta.total_seconds() // 86400)
    
    if delta.total_seconds() < 0:
        return (0.0, days, True) # Expired
    
    if days <= 3: return (100.0, days, False)
    if days <= 7: return (90.0, days, False)
    if days <= 14: return (75.0, days, False)
    if days <= 30: return (50.0, days, False)
    return (25.0, days, False)


def _completeness(ex: OpportunityExtraction) -> float:
    """Logical completeness check (not a heuristic)."""
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


def score_opportunity(
    profile: StudentProfile,
    ex: OpportunityExtraction,
    *,
    base: datetime,
    rag_context: str = "",
) -> tuple[ScoreBreakdown, list[str]]:
    """
    Computes final score using semantic fit from LLM + logical urgency/completeness.
    All heuristics (rapidfuzz, GPA regex, semester checks) have been removed.
    """
    deadline_dt: datetime | None = None
    if ex.deadline_iso:
        try:
            deadline_dt = datetime.fromisoformat(ex.deadline_iso)
            if deadline_dt and deadline_dt.tzinfo is None:
                deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
        except Exception:
            deadline_dt = None
    
    if not deadline_dt and ex.deadline_text:
        deadline_dt = parse_deadline_to_datetime(ex.deadline_text, base=base)

    # 1. Fit (from LLM)
    fit = getattr(ex, "fit_score", 50.0)
    fit_reasons = getattr(ex, "fit_reasons", [])

    # 2. Urgency (mathematical)
    urgency, days, expired = _urgency_score(deadline_dt, base=base)

    # 3. Completeness (logical)
    completeness = _completeness(ex)

    # Final weighted average
    total = clamp(0.50 * fit + 0.35 * urgency + 0.15 * completeness, 0.0, 100.0)

    # Compile reasons for the student
    reasons: list[str] = []
    reasons.extend(fit_reasons[:3]) # Show LLM's semantic fit reasons first
    
    if days is not None:
        if expired:
            reasons.append("Deadline has passed.")
        else:
            reasons.append(f"Deadline in ~{max(days, 0)} days.")
    else:
        reasons.append("No clear deadline (lower urgency).")

    score = ScoreBreakdown(
        fit=fit,
        urgency=urgency,
        completeness=completeness,
        total=total,
        days_to_deadline=days,
        expired=expired,
    )
    return score, reasons
