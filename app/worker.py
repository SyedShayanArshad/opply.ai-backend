import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlmodel import Session, select

from .db import engine
from .db_models import User, DBStudentProfile, DBEmailRecord
from .extract import extract_opportunity
from .mistral_client import MistralLLM
from .models import EmailInput, StudentProfile, FinancialNeedLevel, LocationPreference
from .profile_summary import build_profile_summary
from .record_explanation import build_record_explanation
from .score import score_opportunity
from .rag_engine import build_profile_vectorstore, retrieve_relevant_context
from .utils import now_utc
from .ws_manager import manager
from .twilio_service import send_whatsapp_alert

logger = logging.getLogger("app.worker")
_llm = MistralLLM()


def _decode_gmail_body(data: str | None) -> str:
    if not data:
        return ""
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_text_parts(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []

    texts: list[str] = []
    mime_type = payload.get("mimeType", "")
    body = payload.get("body") or {}

    if mime_type.startswith("text/plain"):
        text = _decode_gmail_body(body.get("data"))
        if text.strip():
            texts.append(text)

    for part in payload.get("parts", []) or []:
        texts.extend(_collect_text_parts(part))

    return texts


def _gmail_headers_to_map(headers: list[dict[str, str]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for header in headers or []:
        name = (header.get("name") or "").lower()
        value = header.get("value") or ""
        if name and value:
            out[name] = value
    return out


def _db_profile_to_pydantic(db_profile: DBStudentProfile) -> StudentProfile:
    """Convert a DB profile row into the Pydantic StudentProfile model."""
    return StudentProfile(
        degree_program=db_profile.degree_program,
        semester=db_profile.semester,
        cgpa=db_profile.cgpa,
        skills=[s.strip() for s in db_profile.skills_csv.split(",") if s.strip()],
        interests=[s.strip() for s in db_profile.interests_csv.split(",") if s.strip()],
        preferred_opportunity_types=[],
        financial_need=FinancialNeedLevel(db_profile.financial_need),
        location_preference=LocationPreference(db_profile.location_preference),
        location_text=db_profile.location_text,
        past_experience=db_profile.past_experience,
        profile_summary=db_profile.profile_summary,
    )


async def _refresh_google_access_token(user: User, session: Session) -> bool:
    if not user.google_refresh_token:
        return False

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        return False

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": user.google_refresh_token,
        "grant_type": "refresh_token",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(token_url, data=payload)
        if resp.status_code != 200:
            return False

        data = resp.json()
        new_access_token = data.get("access_token")
        if not new_access_token:
            return False

        user.google_access_token = new_access_token
        user.token_expiry = int(time.time()) + int(data.get("expires_in", 3600))
        session.add(user)
        session.commit()
        return True
    except Exception:
        return False


async def fetch_unseen_emails(user: User, session: Session, last_sync_date: datetime | None = None) -> list[dict[str, Any]]:
    access_token = user.google_access_token
    if not access_token:
        return []

    now_ts = int(time.time())
    if user.token_expiry and user.token_expiry <= (now_ts + 30):
        refreshed = await _refresh_google_access_token(user, session)
        if not refreshed:
            await manager.send_personal_message({
                "type": "error",
                "message": "Gmail session expired or revoked. Please reconnect your Google account in Settings."
            }, user.id)
            return []
        access_token = user.google_access_token

    if not access_token:
        return []

    start = last_sync_date or datetime.now(timezone.utc) - timedelta(hours=24)
    query = f"is:unread after:{int(start.timestamp())}"
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            list_resp = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                headers=headers,
                params={"q": query, "maxResults": 10},
            )

            if list_resp.status_code == 401:
                refreshed = await _refresh_google_access_token(user, session)
                if not refreshed or not user.google_access_token:
                    return []
                headers["Authorization"] = f"Bearer {user.google_access_token}"
                list_resp = await client.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                    headers=headers,
                    params={"q": query, "maxResults": 10},
                )

            if list_resp.status_code == 403:
                err_msg = list_resp.json().get("error", {}).get("message", "")
                if "disabled" in err_msg.lower():
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Gmail API is disabled in your Google Cloud Project. Please enable it at the URL shown in your Google Cloud Console."
                    }, user.id)
                    logger.error("Gmail API disabled for %s: %s", user.email, err_msg)
                return []
            elif list_resp.status_code != 200:
                logger.error("Gmail API error for %s: %s", user.email, list_resp.text)
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Gmail sync failed: API returned status {list_resp.status_code}. This might be a temporary Google service issue."
                }, user.id)
                return []

            message_refs = list_resp.json().get("messages", []) or []
            emails_fetched: list[dict[str, Any]] = []

            for ref in message_refs:
                message_id = ref.get("id")
                if not message_id:
                    continue

                message_resp = await client.get(
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                    headers=headers,
                    params={"format": "full"},
                )
                if message_resp.status_code != 200:
                    continue

                payload = message_resp.json()
                headers_map = _gmail_headers_to_map(payload.get("payload", {}).get("headers"))
                text_parts = _collect_text_parts(payload.get("payload"))
                body_text = "\n\n".join([t for t in text_parts if t.strip()])

                emails_fetched.append(
                    {
                        "message_id": headers_map.get("message-id", message_id),
                        "parsed": {
                            "subject": headers_map.get("subject", "No Subject"),
                            "sender": headers_map.get("from", "Unknown Sender"),
                            "body": body_text.strip(),
                        },
                        "raw": json.dumps(payload),
                    }
                )

            return emails_fetched
    except Exception as e:
        logger.error("Gmail fetch error for %s: %s", user.email, e)
        await manager.send_personal_message({
            "type": "error",
            "message": f"Gmail connection error: {str(e)}"
        }, user.id)
        return []


async def process_user_emails(user: User, db_profile: DBStudentProfile, session: Session):
    if not user.google_access_token:
        return

    new_emails = await fetch_unseen_emails(user, session, last_sync_date=user.last_sync_date)

    # Always update sync date so we don't look backwards indefinitely
    user.last_sync_date = datetime.now(timezone.utc)
    session.add(user)
    session.commit()

    if not new_emails:
        await manager.send_personal_message({"type": "sync_complete", "message": "Gmail sync complete. No new emails found."}, user.id)
        return

    logger.info("Found %d new emails for %s", len(new_emails), user.email)
    await manager.send_personal_message({"type": "progress", "message": f"Found {len(new_emails)} new emails. Analyzing..."}, user.id)

    profile = _db_profile_to_pydantic(db_profile)
    profile_summary = db_profile.profile_summary or await build_profile_summary(profile, _llm)
    base = now_utc()

    # ── Build per-user RAG profile vector store (cached if profile unchanged) ──
    rag_ready = False
    try:
        rag_ready = await asyncio.to_thread(build_profile_vectorstore, profile, user.id)
        if rag_ready:
            logger.info("RAG profile store ready for user %s", user.email)
        else:
            logger.info("RAG store unavailable for user %s — using plain profile context", user.email)
    except Exception as rag_err:
        logger.warning("RAG store build failed for user %s: %s", user.email, rag_err)

    for em in new_emails:
        message_id = em["message_id"]

        # Check if already processed
        exists = session.exec(select(DBEmailRecord).where(DBEmailRecord.email_id == message_id, DBEmailRecord.user_id == user.id)).first()
        if exists:
            continue

        parsed = em["parsed"]
        email_input = EmailInput(
            id=message_id,
            subject=parsed["subject"],
            sender=parsed["sender"],
            body=parsed["body"],
            raw=em["raw"]
        )

        await manager.send_personal_message({"type": "progress", "message": f"Analyzing: {email_input.subject[:40]}..."}, user.id)

        # ── Retrieve RAG context for this specific email ──────────────────────
        email_query_text = f"{email_input.subject or ''} {email_input.body or ''}"[:1200]
        rag_context = ""
        if rag_ready:
            try:
                rag_context = await asyncio.to_thread(
                    retrieve_relevant_context, email_query_text, user.id, 3
                )
            except Exception as rc_err:
                logger.warning("RAG retrieval failed for email %s: %s", message_id, rc_err)

        # ── Run RAG-augmented extraction ──────────────────────────────────────
        ex = await extract_opportunity(
            email_input,
            profile,
            base=base,
            llm=_llm,
            notice_text=None,
            profile_summary=profile_summary,
            rag_context=rag_context,
            user_id=user.id,
        )

        if not ex.is_opportunity:
            explanation = await build_record_explanation(
                classification="not important",
                profile_summary=profile_summary,
                email=email_input,
                ex=ex,
                llm=_llm,
                score=None,
                rag_context=rag_context,
            )
            record = DBEmailRecord(
                user_id=user.id,
                email_id=message_id,
                subject=email_input.subject or "",
                sender=email_input.sender or "",
                classification="not important",
                explanation=explanation,
                extraction_json=ex.model_dump_json(),
                source="gmail",
            )
        else:
            score, reasons = score_opportunity(profile, ex, base=base, rag_context=rag_context)
            explanation = await build_record_explanation(
                classification="important",
                profile_summary=profile_summary,
                email=email_input,
                ex=ex,
                llm=_llm,
                score=score,
                rag_context=rag_context,
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
                subject=email_input.subject or "",
                sender=email_input.sender or "",
                classification="important",
                explanation=explanation,
                opportunity_type=ex.opportunity_type.value if ex.opportunity_type else None,
                score=score.total,
                summary=ex.summary,
                detailed_actions_json=json.dumps(checklist),
                extraction_json=ex.model_dump_json(),
                score_json=score.model_dump_json(),
                source="gmail",
            )
            
            # WhatsApp Alert Integration
            if user.whatsapp_enabled and user.phone_number:
                steps_text = "\n- ".join(checklist) if checklist else "No specific steps found."
                opp_type = ex.opportunity_type.value.title() if ex.opportunity_type else "Opportunity"
                msg = (
                    f"🌟 *Opply AI: New Opportunity Detected*\n\n"
                    f"Greetings,\n\n"
                    f"A new {opp_type} has been identified that matches your profile.\n\n"
                    f"📌 *Title:* {ex.title or email_input.subject}\n"
                    f"🏢 *Organization:* {ex.organization or email_input.sender}\n"
                    f"🎯 *Match Score:* {score.total:.1f}/10\n"
                )
                if ex.deadline_text:
                    msg += f"⏰ *Deadline:* {ex.deadline_text}\n"
                
                msg += (
                    f"\n*Executive Summary:*\n{ex.summary}\n\n"
                    f"*Recommended Actions:*\n- {steps_text}\n\n"
                    f"To view full details and direct links, please visit your dashboard:\n"
                    f"🔗 https://opply-ai.vercel.app/dashboard\n\n"
                    f"Best regards,\n"
                    f"The Opply AI Team"
                )
                logger.info("Sending WhatsApp alert to %s for email: %s", user.email, email_input.subject)
                wa_result = send_whatsapp_alert(user.phone_number, msg)
                await manager.send_personal_message({
                    "type": "whatsapp_status",
                    "success": wa_result["success"],
                    "error_reason": wa_result["error_reason"],
                    "subject": email_input.subject,
                }, user.id)

        session.add(record)
        session.commit()
        logger.info("Processed email %s for user %s", message_id, user.email)
        await manager.send_personal_message({"type": "new_record"}, user.id)

    await manager.send_personal_message({"type": "sync_complete", "message": "Gmail sync complete."}, user.id)


async def background_worker_loop():
    logger.info("Starting background worker loop...")
    while True:
        try:
            with Session(engine) as session:
                # Cleanup emails older than 7 days to manage DB storage
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                old_emails = session.exec(select(DBEmailRecord).where(DBEmailRecord.created_at < seven_days_ago)).all()
                if old_emails:
                    for old_em in old_emails:
                        session.delete(old_em)
                    session.commit()
                    logger.info("Deleted %d emails older than 7 days.", len(old_emails))

                users = session.exec(select(User)).all()
                for user in users:
                    profile = session.exec(select(DBStudentProfile).where(DBStudentProfile.user_id == user.id)).first()

                    if not profile:
                        # Notify user to set up profile if they have a WS connection
                        await manager.send_personal_message(
                            {"type": "setup_required", "message": "Please complete your profile to enable email classification."},
                            user.id
                        )
                        continue

                    if user.google_access_token:
                        await process_user_emails(user, profile, session)

        except Exception as e:
            logger.error("Background worker error: %s", e)

        await asyncio.sleep(60)  # Run every 60 seconds
