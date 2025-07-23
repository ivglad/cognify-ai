"""
Database session management with trio support.
"""
import logging
from typing import Generator

import trio
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine with trio-compatible configuration
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG,
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_db_async() -> Session:
    """
    Async database session for trio compatibility.
    """
    return SessionLocal()


async def close_db_async(db: Session):
    """
    Close async database session.
    """
    db.close()


# Event listeners for connection management
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set database pragmas on connection."""
    if "postgresql" in settings.DATABASE_URL:
        # PostgreSQL specific settings
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET timezone TO 'UTC'")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log database connection checkout."""
    logger.debug("Database connection checked out")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log database connection checkin."""
    logger.debug("Database connection checked in")


async def init_db():
    """
    Initialize database with trio support.
    """
    logger.info("Initializing database...")
    
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info("Database connection successful")
        
        # Create tables if they don't exist
        # Note: In production, use Alembic migrations instead
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db():
    """
    Close database connections.
    """
    logger.info("Closing database connections...")
    engine.dispose()