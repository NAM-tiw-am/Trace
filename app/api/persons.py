import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.case import Case
from app.models.person import Person
from app.models.photo import PersonPhoto
from app.models.sighting import Sighting
from app.schemas.person import PersonCreate, PersonUpdate, PersonResponse, PhotoResponse
from app.schemas.sighting import SightingResponse
from app.services.storage_service import StorageService
from app.services.face_service import FaceService

router = APIRouter(prefix="/persons", tags=["Persons"])

@router.post("", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
def create_person(person_in: PersonCreate, db: Session = Depends(get_db)):
    """Register a new missing person under a case."""
    case = db.query(Case).filter(Case.id == person_in.case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {person_in.case_id} not found."
        )

    person = Person(**person_in.model_dump())
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

@router.get("", response_model=List[PersonResponse])
def list_persons(case_id: int | None = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List registered missing persons."""
    query = db.query(Person)
    if case_id:
        query = query.filter(Person.case_id == case_id)
    return query.offset(skip).limit(limit).all()

@router.get("/{id}", response_model=PersonResponse)
def get_person(id: int, db: Session = Depends(get_db)):
    """Get missing person details."""
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")
    return person

@router.put("/{id}", response_model=PersonResponse)
def update_person(id: int, person_in: PersonUpdate, db: Session = Depends(get_db)):
    """Update missing person information."""
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    update_data = person_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(person, field, value)

    db.commit()
    db.refresh(person)
    return person

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(id: int, db: Session = Depends(get_db)):
    """Delete a missing person record."""
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    db.delete(person)
    db.commit()
    return None

# --- Photos & Embeddings ---

@router.post("/{id}/photos", response_model=PhotoResponse, status_code=status.HTTP_201_CREATED)
def upload_reference_photo(
    id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a reference photo for a missing person.
    Detects face, generates ArcFace 512-D embedding, and stores it in the database.
    """
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    # Save photo to storage
    filename, file_path = StorageService.save_photo(file)

    photo = PersonPhoto(
        person_id=id,
        filename=filename,
        file_path=file_path,
    )
    db.add(photo)
    db.commit()
    db.refresh(photo)

    # Generate and store embedding
    emb = FaceService.process_and_store_photo(
        db=db,
        person_id=id,
        photo_id=photo.id,
        image_path=file_path,
    )
    if emb is None:
        # Photo was uploaded, but no face was detected
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Photo was saved, but no face could be detected. Please upload a clear frontal face image."
        )

    return photo

@router.get("/{id}/photos", response_model=List[PhotoResponse])
def list_person_photos(id: int, db: Session = Depends(get_db)):
    """List all reference photos for a person."""
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")
    return person.photos

@router.delete("/{id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person_photo(id: int, photo_id: int, db: Session = Depends(get_db)):
    """Delete a reference photo and its associated embeddings."""
    photo = db.query(PersonPhoto).filter(PersonPhoto.id == photo_id, PersonPhoto.person_id == id).first()
    if not photo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")

    if os.path.exists(photo.file_path):
        try:
            os.remove(photo.file_path)
        except OSError:
            pass

    db.delete(photo)
    db.commit()
    return None

# --- Person Sightings ---

@router.get("/{id}/sightings", response_model=List[SightingResponse])
def get_person_sightings(id: int, db: Session = Depends(get_db)):
    """Retrieve all potential sightings for a missing person."""
    person = db.query(Person).filter(Person.id == id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    return db.query(Sighting).filter(Sighting.person_id == id).order_by(Sighting.similarity_score.desc()).all()
