import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Trace - Missing Person Identification System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Database: Default PostgreSQL, fallback SQLite for testing/standalone
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/trace_db"
    )

    # Storage paths
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    PHOTOS_DIR: Path = STORAGE_DIR / "photos"
    VIDEOS_DIR: Path = STORAGE_DIR / "videos"
    SNAPSHOTS_DIR: Path = STORAGE_DIR / "snapshots"

    # AI & Face Matching Parameters
    INSIGHTFACE_MODEL: str = "buffalo_l"
    FACE_DETECTION_SIZE: tuple[int, int] = (640, 640)
    MIN_FACE_CONFIDENCE: float = 0.5
    MATCH_THRESHOLD: float = 0.35
    HIGH_CONFIDENCE_THRESHOLD: float = 0.55
    ALERT_COOLDOWN_SECONDS: float = 5.0
    MOTION_DETECTION_ENABLED: bool = True
    MOTION_PIXEL_THRESHOLD: int = 25
    MOTION_AREA_THRESHOLD: float = 0.005
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()

# Ensure directories exist
for directory in [settings.STORAGE_DIR, settings.PHOTOS_DIR, settings.VIDEOS_DIR, settings.SNAPSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
