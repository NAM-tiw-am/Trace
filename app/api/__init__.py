from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.cases import router as cases_router
from app.api.persons import router as persons_router
from app.api.videos import router as videos_router
from app.api.sightings import router as sightings_router
from app.api.jobs import router as jobs_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(cases_router)
api_router.include_router(persons_router)
api_router.include_router(videos_router)
api_router.include_router(sightings_router)
api_router.include_router(jobs_router)
