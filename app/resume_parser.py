"""
resume_parser.py — Opply AI Resume Drag & Drop Parser

Extracts text from PDF/Docx files and uses Mistral LLM to parse into profile fields.
"""
from __future__ import annotations

import io
import logging
from typing import Any

from .mistral_client import MistralLLM
from .models import StudentProfile

logger = logging.getLogger("app.resume_parser")

_SYSTEM = """\
You are an expert student profile analyzer for Opply AI.
Your Goal:
Extract the student's academic and professional information from the provided resume text and map it to the requested JSON schema.
If a field is not found in the resume, leave it as null, empty string, or empty array according to the type.

Rules:
- Do NOT make up information or extract irrelevant boilerplate text.
- If the resume does NOT contain work or project experience, set past_experience to null.
- Do not extract hobbies or personal statements as past_experience. Only extract real work, internships, or academic projects.
- Return ONLY strict JSON. No markdown.
"""

_SCHEMA_HINT = """\
{
  "degree_program": "string|null (e.g. BS Computer Science)",
  "semester": "integer|null (guess based on graduation year if possible, else 1)",
  "cgpa": "float|null",
  "skills": ["string (e.g. Python, React, Data Analysis)"],
  "interests": ["string (e.g. Machine Learning, Open Source)"],
  "past_experience": "string|null (A summarized paragraph of their work/project experience. Extract ONLY actual job/internship/project history. If none, return null.)"
}"""


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF or Docx bytes."""
    text = ""
    lower_name = filename.lower()
    
    try:
        if lower_name.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
            
        elif lower_name.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text_parts = [para.text for para in doc.paragraphs if para.text]
            text = "\n".join(text_parts)
        else:
            logger.warning("Unsupported file type: %s", filename)
            
    except Exception as exc:
        logger.error("Failed to extract text from %s: %s", filename, exc)
    
    return text.strip()


async def parse_resume_with_llm(resume_text: str, llm: MistralLLM) -> dict[str, Any]:
    """Parse resume text using LLM and return dict of profile fields."""
    if not llm.available or not resume_text:
        return {}

    user_prompt = f"Extract profile fields from this resume:\n\n{resume_text[:4000]}"
    
    try:
        data = await llm.json_extract(
            system=_SYSTEM, user=user_prompt, schema_hint=_SCHEMA_HINT
        )
        return data
    except Exception as exc:
        logger.error("LLM resume parsing failed: %s", exc)
        return {}
