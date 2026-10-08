from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.sighting import Sighting
from app.schemas.sighting import SightingResponse, SightingStatusUpdate

router = APIRouter(prefix="/sightings", tags=["Sightings"])

@router.get("", response_model=List[SightingResponse])
def list_sightings(
    person_id: Optional[int] = None,
    video_id: Optional[int] = None,
    match_status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all sightings with optional filtering by person, video, or review status."""
    query = db.query(Sighting)
    if person_id is not None:
        query = query.filter(Sighting.person_id == person_id)
    if video_id is not None:
        query = query.filter(Sighting.video_id == video_id)
    if match_status is not None:
        query = query.filter(Sighting.match_status == match_status)

    return query.order_by(Sighting.created_at.desc()).offset(skip).limit(limit).all()

@router.get("/{id}", response_model=SightingResponse)
def get_sighting(id: int, db: Session = Depends(get_db)):
    """Get sighting details."""
    sighting = db.query(Sighting).filter(Sighting.id == id).first()
    if not sighting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sighting not found.")
    return sighting

@router.put("/{id}/status", response_model=SightingResponse)
def update_sighting_status(
    id: int,
    status_update: SightingStatusUpdate,
    db: Session = Depends(get_db),
):
    """
    Update sighting verification status by investigator.
    Allowed statuses: POTENTIAL_MATCH, REVIEWED, CONFIRMED, REJECTED.
    """
    sighting = db.query(Sighting).filter(Sighting.id == id).first()
    if not sighting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sighting not found.")

    valid_statuses = {"POTENTIAL_MATCH", "REVIEWED", "CONFIRMED", "REJECTED"}
    if status_update.match_status.upper() not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    sighting.match_status = status_update.match_status.upper()
    if status_update.notes is not None:
        sighting.notes = status_update.notes

    db.commit()
    db.refresh(sighting)
    return sighting

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sighting(id: int, db: Session = Depends(get_db)):
    """Delete a sighting."""
    sighting = db.query(Sighting).filter(Sighting.id == id).first()
    if not sighting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sighting not found.")

    db.delete(sighting)
    db.commit()
    return None
