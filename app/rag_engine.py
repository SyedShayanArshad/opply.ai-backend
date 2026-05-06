"""
rag_engine.py — Opply AI RAG Engine

Per-user FAISS in-memory vector store backed by Mistral embeddings (free tier).
Profile is split into semantic chunks and embedded once per worker cycle.
Each email query retrieves the most relevant profile context before LLM extraction.

Memory footprint on Render free tier:
  - faiss-cpu: ~15 MB
  - langchain-mistralai embeddings: API-based, zero extra RAM
  - FAISS index per user (~10 chunks × 1024-dim float32): < 0.5 MB per user
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("app.rag_engine")

# ── In-memory stores ───────────────────────────────────────────────────────────
# { user_id: {"store": FAISS, "profile_hash": str} }
_user_stores: dict[int, dict[str, Any]] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _profile_hash(profile) -> str:
    """Stable hash of a StudentProfile so we skip rebuilding unchanged stores."""
    key = (
        profile.degree_program,
        profile.semester,
        profile.cgpa,
        tuple(sorted(profile.skills)),
        tuple(sorted(profile.interests)),
        tuple(sorted(t.value for t in profile.preferred_opportunity_types)),
        profile.financial_need.value,
        profile.location_preference.value,
        profile.location_text or "",
        profile.past_experience or "",
        profile.profile_summary or "",
    )
    return hashlib.md5(str(key).encode()).hexdigest()


def _profile_to_chunks(profile) -> list[str]:
    """
    Split a StudentProfile into small text chunks for embedding.
    Each chunk covers one semantic facet of the student.
    """
    chunks: list[str] = []

    if profile.degree_program:
        chunks.append(
            f"Student studies {profile.degree_program}, currently in semester "
            f"{profile.semester} out of a typical 8-semester program."
        )

    chunks.append(
        f"Academic performance: CGPA {profile.cgpa:.2f} out of 4.0. "
        f"Financial need level: {profile.financial_need.value}."
    )

    if profile.skills:
        chunks.append(f"Technical and professional skills: {', '.join(profile.skills)}.")

    if profile.interests:
        chunks.append(f"Academic and career interests: {', '.join(profile.interests)}.")

    if profile.preferred_opportunity_types:
        types_str = ", ".join(t.value for t in profile.preferred_opportunity_types)
        chunks.append(f"Preferred opportunity types the student wants: {types_str}.")

    loc = profile.location_text or profile.location_preference.value
    chunks.append(f"Location preference for opportunities: {loc}.")

    if profile.past_experience:
        chunks.append(
            f"Past experience, internships, and projects: {profile.past_experience}"
        )

    if profile.profile_summary:
        chunks.append(f"Overall student summary: {profile.profile_summary}")

    return [c.strip() for c in chunks if c.strip()]


def _get_embeddings():
    """
    Return a MistralAIEmbeddings instance (uses Mistral free-tier API).
    Returns None if the API key is missing or langchain-mistralai is not installed.
    """
    import os
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        logger.warning("RAG: MISTRAL_API_KEY not set — vector store disabled.")
        return None
    try:
        from langchain_mistralai import MistralAIEmbeddings  # type: ignore
        return MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    except Exception as exc:
        logger.warning("RAG: Could not init MistralAIEmbeddings: %s", exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def build_profile_vectorstore(profile, user_id: int) -> bool:
    """
    Build (or reuse if unchanged) a per-user FAISS vector store.

    Returns True  → store ready, RAG is active for this user.
    Returns False → embedding unavailable, caller falls back to plain text profile.
    """
    current_hash = _profile_hash(profile)
    existing = _user_stores.get(user_id)
    if existing and existing.get("profile_hash") == current_hash:
        logger.debug("RAG: Reusing existing store for user %d (profile unchanged).", user_id)
        return True

    embeddings = _get_embeddings()
    if embeddings is None:
        return False

    chunks = _profile_to_chunks(profile)
    if not chunks:
        return False

    try:
        from langchain_core.documents import Document  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore

        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = FAISS.from_documents(docs, embeddings)

        _user_stores[user_id] = {
            "store": vectorstore,
            "profile_hash": current_hash,
        }
        logger.info(
            "RAG: Built FAISS store for user %d — %d profile chunks indexed.",
            user_id,
            len(chunks),
        )
        return True

    except Exception as exc:
        logger.warning("RAG: Failed to build vector store for user %d: %s", user_id, exc)
        return False


def retrieve_relevant_context(email_text: str, user_id: int, k: int = 3) -> str:
    """
    Retrieve the top-k most semantically relevant profile chunks for a given email.

    Returns a formatted string to inject into the LLM prompt, or "" on failure.
    The query uses the first 800 chars of the email to keep embedding cost low.
    """
    entry = _user_stores.get(user_id)
    if not entry:
        return ""

    vectorstore = entry.get("store")
    if not vectorstore:
        return ""

    try:
        query = email_text[:800].strip()
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""

        context_lines = [doc.page_content for doc in docs]
        return "\n".join(context_lines)

    except Exception as exc:
        logger.warning("RAG: Context retrieval failed for user %d: %s", user_id, exc)
        return ""


def clear_user_store(user_id: int) -> None:
    """Remove a user's vector store (call after profile update)."""
    _user_stores.pop(user_id, None)
    logger.debug("RAG: Cleared store for user %d.", user_id)
