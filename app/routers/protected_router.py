from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from ..db import get_session
from ..db_models import User, DBStudentProfile, DBEmailRecord
from ..routers.auth_router import get_current_user
from ..models import StudentProfile, EmailInput, AnalyzeResponse, RankedOpportunity, ScoreBreakdown, OpportunityExtraction, EmailRecord
from ..extract import extract_opportunity
from ..score import score_opportunity
from ..mistral_client import MistralLLM
from ..profile_summary import build_profile_summary
from ..record_explanation import build_record_explanation
from ..worker import process_user_emails
from ..utils import now_utc
import json
import hashlib
import logging
from pydantic import BaseModel
from typing import Optional
from ..ws_manager import manager
from ..twilio_service import send_whatsapp_alert

logger = logging.getLogger("app.protected")

router = APIRouter(prefix="/api/protected", tags=["protected"])
_llm = MistralLLM()


# ─── Helpers ────────────────────────────────────────────────────────

def _db_profile_to_pydantic(db_profile: DBStudentProfile) -> StudentProfile:
    """Convert a DB profile row into the Pydantic StudentProfile model."""
    return StudentProfile(
        degree_program=db_profile.degree_program,
        semester=db_profile.semester,
        cgpa=db_profile.cgpa,
        skills=[s.strip() for s in db_profile.skills_csv.split(",") if s.strip()],
        interests=[s.strip() for s in db_profile.interests_csv.split(",") if s.strip()],
        preferred_opportunity_types=[x for x in db_profile.preferred_opportunity_types_csv.split(",") if x.strip()],
        financial_need=db_profile.financial_need,
        location_preference=db_profile.location_preference,
        location_text=db_profile.location_text,
        past_experience=db_profile.past_experience,
        profile_summary=db_profile.profile_summary,
    )


def _stable_email_id(subject: str, sender: str, body: str) -> str:
    """Generate a deterministic email ID from content to avoid duplicates."""
    raw = f"{(subject or '').strip()}|{(sender or '').strip()}|{(body or '').strip()[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ─── Onboarding Status ─────────────────────────────────────────────

class OnboardingStatusResponse(BaseModel):
    has_profile: bool
    has_oauth: bool
    profile_summary: Optional[str] = None
    email_count: int = 0
    connected_email: Optional[str] = None


@router.get("/onboarding-status", response_model=OnboardingStatusResponse)
def get_onboarding_status(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    email_count = len(session.exec(select(DBEmailRecord).where(DBEmailRecord.user_id == user.id)).all())

    return OnboardingStatusResponse(
        has_profile=db_profile is not None,
        has_oauth=bool(user.google_access_token),
        profile_summary=db_profile.profile_summary if db_profile else None,
        email_count=email_count,
        connected_email=user.connected_email,
    )


# ─── Profile ───────────────────────────────────────────────────────

@router.get("/profile", response_model=StudentProfile)
def get_profile(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    if not db_profile:
        # Return default
        return StudentProfile(degree_program="", semester=1, cgpa=0.0)

    return _db_profile_to_pydantic(db_profile)


@router.post("/profile")
async def update_profile(profile: StudentProfile, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    if not db_profile:
        db_profile = DBStudentProfile(user_id=user.id, degree_program="")
        session.add(db_profile)

    db_profile.degree_program = profile.degree_program
    db_profile.semester = profile.semester
    db_profile.cgpa = profile.cgpa
    db_profile.skills_csv = ",".join(profile.skills)
    db_profile.interests_csv = ",".join(profile.interests)
    db_profile.preferred_opportunity_types_csv = ",".join([p.value if hasattr(p, 'value') else str(p) for p in profile.preferred_opportunity_types])
    db_profile.financial_need = profile.financial_need.value if hasattr(profile.financial_need, 'value') else profile.financial_need
    db_profile.location_preference = profile.location_preference.value if hasattr(profile.location_preference, 'value') else profile.location_preference
    db_profile.location_text = profile.location_text
    db_profile.past_experience = profile.past_experience
    db_profile.profile_summary = await build_profile_summary(profile, _llm)

    session.commit()
    return {"status": "ok", "profile_summary": db_profile.profile_summary}


# ─── Dashboard ─────────────────────────────────────────────────────

@router.get("/dashboard", response_model=AnalyzeResponse)
def get_dashboard(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    records = session.exec(select(DBEmailRecord).where(DBEmailRecord.user_id == user.id)).all()

    ranked = []
    discarded = []
    email_records = []

    for r in records:
        ex_data = json.loads(r.extraction_json) if r.extraction_json else {}
        er = EmailRecord(
            email_id=r.email_id,
            subject=r.subject,
            sender=r.sender,
            classification=r.classification,
            explanation=r.explanation,
            opportunity_type=r.opportunity_type,
            score=r.score,
            summary=r.summary,
            detailed_actions=json.loads(r.detailed_actions_json) if r.detailed_actions_json else [],
            links=ex_data.get("links", []) if isinstance(ex_data, dict) else [],
            source=getattr(r, "source", "manual"),
            created_at=r.created_at.isoformat() if r.created_at else None,
            email_date=r.email_date.isoformat() if getattr(r, "email_date", None) else None,
        )
        email_records.append(er)

        if r.classification == "important":
            score_data = json.loads(r.score_json) if r.score_json else {"total": r.score or 0.0, "fit": 0.0, "urgency": 0.0, "completeness": 0.0}

            ex = OpportunityExtraction(**ex_data)
            sc = ScoreBreakdown(**score_data)

            ranked.append(RankedOpportunity(
                email_id=r.email_id,
                subject=r.subject,
                sender=r.sender,
                extracted=ex,
                score=sc,
                reasons=[r.explanation],
                action_checklist=er.detailed_actions
            ))
        else:
            discarded.append({
                "email_id": r.email_id,
                "subject": r.subject,
                "sender": r.sender,
                "reason": r.explanation
            })

    ranked.sort(key=lambda x: x.score.total, reverse=True)

    # Check profile status for meta
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()

    return AnalyzeResponse(
        ranked=ranked,
        discarded=discarded,
        email_records=email_records,
        meta={
            "important": len(ranked),
            "not_important": len(discarded),
            "total_emails": len(records),
            "oauth_connected": bool(user.google_access_token),
            "connected_email": user.connected_email,
            "has_profile": db_profile is not None,
        }
    )


# ─── Manual Batch Analysis ─────────────────────────────────────────

class ManualAnalyzeRequest(BaseModel):
    emails: list[EmailInput]


@router.post("/analyze-manual", response_model=dict)
async def analyze_manual(req: ManualAnalyzeRequest, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    if not db_profile:
        raise HTTPException(status_code=400, detail="Please update your profile first")

    profile = _db_profile_to_pydantic(db_profile)
    profile_summary = db_profile.profile_summary or await build_profile_summary(profile, _llm)
    base = now_utc()
    processed = 0

    await manager.send_personal_message({"type": "progress", "message": f"Manual analysis: Started ({len(req.emails)} emails)..."}, user.id)

    for em in req.emails:
        message_id = em.id or _stable_email_id(em.subject or "", em.sender or "", em.body or em.raw or "")

        # Avoid dupes
        exists = session.exec(select(DBEmailRecord).where(DBEmailRecord.email_id == message_id, DBEmailRecord.user_id == user.id)).first()
        if exists:
            continue

        await manager.send_personal_message({"type": "progress", "message": f"Analyzing: {(em.subject or 'email')[:40]}..."}, user.id)

        ex = await extract_opportunity(
            em,
            profile,
            base=base,
            llm=_llm,
            notice_text=None,
            profile_summary=profile_summary
        )

        if not ex.is_opportunity:
            explanation = await build_record_explanation(
                classification="not important",
                profile_summary=profile_summary,
                email=em,
                ex=ex,
                llm=_llm,
                score=None,
            )
            record = DBEmailRecord(
                user_id=user.id,
                email_id=message_id,
                subject=em.subject or "",
                sender=em.sender or "",
                classification="not important",
                explanation=explanation,
                extraction_json=ex.model_dump_json(),
                source="manual",
            )
        else:
            score, reasons = score_opportunity(profile, ex, base=base)
            explanation = await build_record_explanation(
                classification="important",
                profile_summary=profile_summary,
                email=em,
                ex=ex,
                llm=_llm,
                score=score,
            )

            checklist = []
            checklist.extend(ex.next_steps[:4])
            if ex.required_documents:
                checklist.append("Gather required documents: " + ", ".join(ex.required_documents[:6]))
            if ex.links:
                checklist.append("Apply / learn more: " + ex.links[0])

            record = DBEmailRecord(
                user_id=user.id,
                email_id=message_id,
                subject=em.subject or "",
                sender=em.sender or "",
                classification="important",
                explanation=explanation,
                opportunity_type=ex.opportunity_type.value if ex.opportunity_type else None,
                score=score.total,
                summary=ex.summary,
                detailed_actions_json=json.dumps(checklist),
                extraction_json=ex.model_dump_json(),
                score_json=score.model_dump_json(),
                source="manual",
            )
            
            # WhatsApp Alert Integration for manual analysis
            if user.whatsapp_enabled and user.phone_number:
                steps_text = "\n- ".join(checklist) if checklist else "No specific steps found."
                opp_type = ex.opportunity_type.value.title() if ex.opportunity_type else "Opportunity"
                msg = (
                    f"🚀 *New {opp_type} Found!*\n\n"
                    f"*Subject:* {em.subject}\n"
                    f"*From:* {em.sender}\n"
                    f"*Score:* {score.total:.1f}/10\n\n"
                    f"*Summary:*\n{ex.summary}\n\n"
                    f"*Next Steps:*\n- {steps_text}\n\n"
                    f"📱 Check your Opply AI dashboard for full details!"
                )
                logger.info("Sending WhatsApp alert to %s from manual analysis", user.email)
                send_whatsapp_alert(user.phone_number, msg)

        session.add(record)
        session.commit()
        processed += 1
        await manager.send_personal_message({"type": "new_record"}, user.id)

    await manager.send_personal_message({"type": "progress", "message": "Manual analysis complete."}, user.id)
    return {"status": "ok", "processed": processed}


# ─── Single Email Analysis ─────────────────────────────────────────

class SingleEmailRequest(BaseModel):
    subject: Optional[str] = None
    sender: Optional[str] = None
    body: str


@router.post("/analyze-single", response_model=dict)
async def analyze_single(req: SingleEmailRequest, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Convenience endpoint: paste a single email for classification."""
    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    if not db_profile:
        raise HTTPException(status_code=400, detail="Please update your profile first")

    email_input = EmailInput(
        id=_stable_email_id(req.subject or "", req.sender or "", req.body),
        subject=req.subject,
        sender=req.sender,
        body=req.body,
    )

    # Delegate to the batch handler with a single email
    batch_req = ManualAnalyzeRequest(emails=[email_input])
    return await analyze_manual(batch_req, user=user, session=session)


# ─── Email Deletion ────────────────────────────────────────────────

@router.delete("/emails/{email_id}")
def delete_email(email_id: str, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    record = session.exec(
        select(DBEmailRecord).where(DBEmailRecord.email_id == email_id, DBEmailRecord.user_id == user.id)
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Email record not found")
    session.delete(record)
    session.commit()
    return {"status": "ok"}


@router.delete("/emails")
def delete_all_emails(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    records = session.exec(select(DBEmailRecord).where(DBEmailRecord.user_id == user.id)).all()
    for r in records:
        session.delete(r)
    session.commit()
    return {"status": "ok", "deleted": len(records)}


# ─── Account & OAuth Management ────────────────────────────────────

@router.delete("/oauth/google")
def disconnect_google_oauth(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Disconnect Google OAuth connection"""
    user.google_access_token = None
    user.google_refresh_token = None
    user.token_expiry = None
    user.connected_email = None
    user.last_sync_date = None
    
    session.add(user)
    session.commit()
    return {"status": "ok", "message": "Google account disconnected"}

class AccountResponse(BaseModel):
    email: str
    connected_email: Optional[str] = None
    has_oauth: bool
    phone_number: Optional[str] = None
    whatsapp_enabled: bool = False

@router.get("/account", response_model=AccountResponse)
def get_account_details(user: User = Depends(get_current_user)):
    """Get user account details"""
    return AccountResponse(
        email=user.email,
        connected_email=user.connected_email,
        has_oauth=bool(user.google_access_token),
        phone_number=user.phone_number,
        whatsapp_enabled=user.whatsapp_enabled
    )

class UpdateAccountRequest(BaseModel):
    new_password: Optional[str] = None
    phone_number: Optional[str] = None
    whatsapp_enabled: Optional[bool] = None

@router.put("/account")
def update_account_details(req: UpdateAccountRequest, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Update user account details"""
    from ..auth import get_password_hash
    updated = False
    if req.new_password:
        user.hashed_password = get_password_hash(req.new_password)
        updated = True
    if req.phone_number is not None:
        # Normalize: strip spaces, ensure it starts with +
        phone = req.phone_number.strip().replace(" ", "")
        if phone and not phone.startswith("+"):
            phone = "+" + phone
        user.phone_number = phone or None
        updated = True
    if req.whatsapp_enabled is not None:
        user.whatsapp_enabled = req.whatsapp_enabled
        updated = True
        
    if updated:
        session.add(user)
        session.commit()
        return {"status": "ok", "message": "Account updated successfully"}
    raise HTTPException(status_code=400, detail="Invalid request - no fields to update")


# ─── Test WhatsApp ─────────────────────────────────────────────────

@router.post("/test-whatsapp")
def test_whatsapp_alert(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Send a test WhatsApp message to confirm Twilio sandbox is set up correctly."""
    if not user.phone_number:
        raise HTTPException(status_code=400, detail="No phone number saved. Please add your WhatsApp number first.")
    
    from twilio.rest import Client
    import os
    sid_val = os.getenv("TWILIO_ACCOUNT_SID")
    token_val = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

    if not sid_val or not token_val or not from_number:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured in .env file.")

    to_number = user.phone_number
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"

    msg = (
        f"🧪 *Opply AI – Test Message*\n\n"
        f"Hello! This is a test notification from Opply AI.\n\n"
        f"✅ Your WhatsApp notifications are working correctly!\n\n"
        f"You will receive messages like this whenever a high-priority opportunity email is detected."
    )
    try:
        client = Client(sid_val, token_val)
        message = client.messages.create(from_=from_number, body=msg, to=to_number)
        return {"status": "ok", "message": f"Test message sent! Check your WhatsApp. (SID: {message.sid})"}
    except Exception as e:
        err_str = str(e)
        if "63038" in err_str or "daily messages limit" in err_str:
            raise HTTPException(
                status_code=429,
                detail="🚦 Daily limit reached: Twilio Sandbox allows only 5 WhatsApp messages per day. Your setup IS working correctly — limit resets at midnight UTC (5:00 AM Pakistan time). Try again tomorrow, or upgrade your Twilio account."
            )
        elif "21608" in err_str or "not verified" in err_str.lower():
            raise HTTPException(
                status_code=400,
                detail="Your number has not joined the Twilio Sandbox. Please send 'join up-lonyer' to +14155238886 on WhatsApp first."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Twilio error: {err_str[:300]}")


# ─── Manual Sync ──────────────────────────────────────────────────

@router.post("/sync")
async def trigger_manual_sync(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Manually trigger a Gmail sync for the current user."""
    if not user.google_access_token:
        raise HTTPException(status_code=400, detail="Google account not connected. Please connect your Gmail first.")

    db_profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()
    if not db_profile:
        raise HTTPException(status_code=400, detail="Profile incomplete. Please set up your academic profile first.")

    # We call the same worker function
    try:
        await process_user_emails(user, db_profile, session)
        return {"status": "ok", "message": "Sync started"}
    except Exception as e:
        logger.error("Manual sync error for %s: %s", user.email, e)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
