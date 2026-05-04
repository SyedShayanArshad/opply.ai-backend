import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import text, inspect

# Load .env file
load_dotenv()

# 1. Setup Database URL (Fallback to SQLite for local dev)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # Fix for Render/Heroku which often provides 'postgres://' instead of 'postgresql://'
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    sqlite_file_name = "database.db"
    DATABASE_URL = f"sqlite:///{sqlite_file_name}"

# 2. Create Engine
# Note: connect_args={"check_same_thread": False} is only needed for SQLite
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    db_type = "PostgreSQL (Neon)" if not DATABASE_URL.startswith("sqlite") else "SQLite (Local)"
    print(f"🚀 Initializing Database: {db_type}")
    SQLModel.metadata.create_all(engine)
    _run_migrations()
    print(f"✅ Database {db_type} is ready!")

def _run_migrations():
    """Run lightweight migrations. Handled differently for SQLite vs PostgreSQL."""
    inspector = inspect(engine)
    
    with engine.begin() as conn:
        # Check User table columns
        user_columns = [col["name"] for col in inspector.get_columns("user")]
        
        if "connected_email" not in user_columns:
            conn.execute(text("ALTER TABLE \"user\" ADD COLUMN connected_email TEXT"))
        
        if "phone_number" not in user_columns:
            conn.execute(text("ALTER TABLE \"user\" ADD COLUMN phone_number TEXT"))
            
        if "whatsapp_enabled" not in user_columns:
            # PostgreSQL needs a type for the added column
            type_bool = "BOOLEAN" if not DATABASE_URL.startswith("sqlite") else "INTEGER"
            conn.execute(text(f"ALTER TABLE \"user\" ADD COLUMN whatsapp_enabled {type_bool} DEFAULT '0'"))

        # Check DBEmailRecord table columns
        email_columns = [col["name"] for col in inspector.get_columns("dbemailrecord")]
        
        if "source" not in email_columns:
            conn.execute(text("ALTER TABLE dbemailrecord ADD COLUMN source TEXT DEFAULT 'manual'"))

def get_session():
    with Session(engine) as session:
        yield session
