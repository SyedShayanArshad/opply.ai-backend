import asyncio
from sqlmodel import Session, create_engine, select
from app.db import engine
from app.db_models import User, DBStudentProfile
from app.routers.protected_router import analyze_manual, ManualAnalyzeRequest
from app.models import EmailInput

async def test():
    with Session(engine) as session:
        user = session.exec(select(User)).first()
        if not user:
            print("No user found")
            return
            
        em = EmailInput(id="test1234", subject="Test Email", sender="test@google.com", body="This is an internship at Google. Deadline is tomorrow. Apply at google.com/careers")
        req = ManualAnalyzeRequest(emails=[em])
        
        try:
            res = await analyze_manual(req, user, session)
            print("Success:", res)
        except Exception as e:
            print("Exception in analyze_manual:", type(e), e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
