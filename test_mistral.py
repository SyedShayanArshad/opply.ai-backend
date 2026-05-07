import asyncio
from app.mistral_client import MistralLLM
from app.models import OpportunityExtraction
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    llm = MistralLLM()
    print("LLM Available:", llm.available)
    system = "Extract a dummy opportunity"
    user = "Subject: Scholarship for CS students. Apply by tomorrow at google.com"
    try:
        res = await llm.structured_extract(system=system, user=user, schema=OpportunityExtraction)
        print("Success:", res)
    except Exception as e:
        print("Error:", type(e), e)

if __name__ == "__main__":
    asyncio.run(main())
