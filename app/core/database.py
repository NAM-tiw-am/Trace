import logging
import json
from sqlalchemy import create_engine, text, TypeDecorator, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

logger = logging.getLogger(__name__)

# Custom JSON-backed Vector type for SQLite fallback
class SQLiteVector(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        import numpy as np
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

# Determine if we can connect to configured PostgreSQL or fallback to SQLite
db_url = settings.DATABASE_URL
engine = None
is_sqlite = False

try:
    if "postgresql" in db_url:
        temp_engine = create_engine(db_url, pool_pre_ping=True)
        with temp_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine = temp_engine
        logger.info(f"Connected to PostgreSQL database: {db_url}")
    else:
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        is_sqlite = True
except Exception as e:
    logger.warning(
        f"Could not connect to PostgreSQL ({e}). "
        f"Falling back to local SQLite database for local development/testing."
    )
    sqlite_path = settings.STORAGE_DIR / "trace.db"
    db_url = f"sqlite:///{sqlite_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    is_sqlite = True

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_vector_type(dim: int = 512):
    """Return pgvector Vector if postgres, else SQLiteVector fallback."""
    if is_sqlite:
        return SQLiteVector()
    try:
        from pgvector.sqlalchemy import Vector
        return Vector(dim)
    except ImportError:
        return SQLiteVector()

def init_db():
    """Initialize database tables and extensions."""
    if not is_sqlite:
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension ensured in PostgreSQL.")
        except Exception as e:
            logger.warning(f"Could not initialize pgvector extension: {e}")

    # Import models so they are registered with Base metadata
    import app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
