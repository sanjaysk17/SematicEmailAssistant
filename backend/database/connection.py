"""
backend/database/connection.py - Database Connection and Session Factory
Manages SQLAlchemy engine, sessionmaker, and table initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

# SQLite compatibility arguments
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


from contextlib import contextmanager


def get_db():
    """Dependency for obtaining a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for obtaining a database session in standalone scripts."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def init_db():
    """Create all database tables."""
    import backend.models.db_models  # noqa: F401
    Base.metadata.create_all(bind=engine)
