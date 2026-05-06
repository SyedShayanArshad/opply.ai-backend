from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class OpportunityType(str, Enum):
    scholarship = "scholarship"
    internship = "internship"
    competition = "competition"
    admissions = "admissions"
    fellowship = "fellowship"
    event = "event"
    other = "other"


class FinancialNeedLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class LocationPreference(str, Enum):
    any = "any"
    local = "local"
    remote = "remote"


class StudentProfile(BaseModel):
    degree_program: str = Field(..., description="e.g., BS Computer Science")
    semester: int = Field(..., ge=1, le=20)
    cgpa: float = Field(..., ge=0.0, le=4.0)
    skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    preferred_opportunity_types: list[OpportunityType] = Field(default_factory=list)
    financial_need: FinancialNeedLevel = FinancialNeedLevel.medium
    location_preference: LocationPreference = LocationPreference.any
    location_text: str | None = Field(
        default=None, description="Optional free-text location, e.g. Lahore / Pakistan"
    )
    past_experience: str | None = Field(
        default=None, description="Short text: internships/projects/positions"
    )
    profile_summary: str | None = Field(
        default=None,
        description="Optional student introduction/summary used as additional context for extraction and ranking.",
    )


class EmailInput(BaseModel):
    id: str | None = None
    subject: str | None = None
    sender: str | None = None
    received_at: datetime | None = None
    body: str | None = None
    raw: str | None = Field(
        default=None,
        description="If provided, the backend will treat this as the full email text.",
    )


class OpportunityExtraction(BaseModel):
    is_opportunity: bool
    opportunity_type: OpportunityType | None = None
    title: str | None = None
    organization: str | None = None
    summary: str | None = None
    deadline_text: str | None = None
    deadline_iso: str | None = None
    location: str | None = None
    eligibility: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    contact: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    evidence: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map field -> short supporting quotes from the email",
    )
    extraction_warnings: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    fit: float = Field(..., ge=0.0, le=100.0)
    urgency: float = Field(..., ge=0.0, le=100.0)
    completeness: float = Field(..., ge=0.0, le=100.0)
    total: float = Field(..., ge=0.0, le=100.0)
    days_to_deadline: int | None = None
    expired: bool = False


class RankedOpportunity(BaseModel):
    email_id: str | None = None
    subject: str | None = None
    sender: str | None = None
    extracted: OpportunityExtraction
    score: ScoreBreakdown
    reasons: list[str] = Field(default_factory=list)
    action_checklist: list[str] = Field(default_factory=list)


class EmailRecord(BaseModel):
    email_id: str | None = None
    subject: str | None = None
    sender: str | None = None
    classification: Literal["important", "not important"]
    explanation: str
    opportunity_type: OpportunityType | None = None
    score: float | None = None
    summary: str | None = None
    detailed_actions: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    source: str = "manual"
    created_at: str | None = None


class AnalyzeRequest(BaseModel):
    student_profile: StudentProfile
    emails: list[EmailInput] = Field(..., min_length=5, max_length=15)
    notice_text: str | None = None
    now_iso: str | None = Field(
        default=None,
        description="Optional override for 'now' in ISO format; useful for demos/tests.",
    )


class ProfileSummaryRequest(BaseModel):
    student_profile: StudentProfile


class ProfileSummaryResponse(BaseModel):
    profile_summary: str


class AnalyzeResponse(BaseModel):
    ranked: list[RankedOpportunity]
    discarded: list[dict[str, Any]] = Field(default_factory=list)
    email_records: list[EmailRecord] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    hint: str | None = None
