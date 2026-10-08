import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.database import init_db
from app.api import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trace")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database tables & extension
    logger.info("Starting up %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    init_db()
    logger.info("Trace backend ready.")
    yield
    logger.info("Shutting down Trace backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered Missing-Person Identification System using Face Recognition in CCTV Footage.",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount storage for direct static access to photos, snapshots, and videos
app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")

# Mount frontend static directory if exists
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "message": "Welcome to Trace API",
            "docs": "/docs",
            "health": f"{settings.API_V1_STR}/health",
        }
