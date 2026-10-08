import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings

class StorageService:
    @staticmethod
    def get_abs_path(path_str: str | Path) -> Path:
        """Resolve path to absolute Path."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (settings.PROJECT_ROOT / p).resolve()

    @staticmethod
    def to_relative_storage_path(path_str: str | Path) -> str:
        """Convert any path to a web-accessible 'storage/...' relative path."""
        s = str(path_str).replace("\\", "/")
        idx = s.find("storage/")
        if idx != -1:
            return s[idx:]
        return s

    @staticmethod
    def save_photo(upload_file: UploadFile) -> tuple[str, str]:
        """Save uploaded photo and return (filename, relative_web_path)."""
        ext = Path(upload_file.filename).suffix or ".jpg"
        unique_name = f"photo_{uuid.uuid4().hex[:12]}{ext}"
        abs_destination = settings.PHOTOS_DIR / unique_name

        with open(abs_destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        relative_path = f"storage/photos/{unique_name}"
        return unique_name, relative_path

    @staticmethod
    def save_video(upload_file: UploadFile) -> tuple[str, str]:
        """Save uploaded CCTV video and return (filename, relative_web_path)."""
        ext = Path(upload_file.filename).suffix or ".mp4"
        unique_name = f"video_{uuid.uuid4().hex[:12]}{ext}"
        abs_destination = settings.VIDEOS_DIR / unique_name

        with open(abs_destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        relative_path = f"storage/videos/{unique_name}"
        return unique_name, relative_path

    @staticmethod
    def get_snapshot_paths(video_id: int, frame_idx: int) -> tuple[Path, str]:
        """Generate (abs_file_path, relative_web_path) for saving a sighting snapshot."""
        filename = f"sighting_v{video_id}_f{frame_idx}_{uuid.uuid4().hex[:8]}.jpg"
        abs_path = settings.SNAPSHOTS_DIR / filename
        relative_path = f"storage/snapshots/{filename}"
        return abs_path, relative_path
