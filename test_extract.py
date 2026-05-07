import asyncio
from app.mistral_client import MistralLLM
from app.models import OpportunityExtraction, EmailInput, StudentProfile
from app.extract import extract_opportunity
from datetime import datetime

async def test():
    llm = MistralLLM()
    em = EmailInput(subject="Test", body="Test")
    prof = StudentProfile(degree_program="BS", semester=1, cgpa=3.0)
    print("Extracting...")
    try:
        res = await extract_opportunity(em, prof, base=datetime.now(), llm=llm)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
