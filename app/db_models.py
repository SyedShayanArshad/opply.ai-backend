import json
from typing import Optional
from datetime import datetime, timezone
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    connected_email: Optional[str] = None
    google_access_token: Optional[str] = None
    google_refresh_token: Optional[str] = None
    token_expiry: Optional[int] = None
    last_sync_date: Optional[datetime] = None
    phone_number: Optional[str] = None
    whatsapp_enabled: bool = Field(default=False)


class DBStudentProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    degree_program: str
    semester: int = 1
    cgpa: float = 0.0
    skills_csv: str = ""
    interests_csv: str = ""
    preferred_opportunity_types_csv: str = ""
    financial_need: str = "medium"
    location_preference: str = "any"
    location_text: Optional[str] = None
    past_experience: Optional[str] = None
    profile_summary: Optional[str] = None


class DBEmailRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    email_id: str = Field(index=True)  # Message identifier from source mailbox to prevent duplicates
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    email_date: Optional[datetime] = None
    subject: str = ""
    sender: str = ""
    classification: str = ""  # "important" or "not important"
    explanation: str = ""
    opportunity_type: Optional[str] = None
    score: Optional[float] = None
    summary: Optional[str] = None
    detailed_actions_json: str = "[]"  # store list[str] as JSON string
    source: str = "manual"  # "gmail" or "manual"

    # Let's also store the raw extraction and score breakdown just in case
    extraction_json: Optional[str] = None
    score_json: Optional[str] = None
