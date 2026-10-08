from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class PhotoResponse(BaseModel):
    id: int
    person_id: int
    filename: str
    file_path: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class EmbeddingResponse(BaseModel):
    id: int
    person_id: int
    photo_id: Optional[int] = None
    model_name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PersonBase(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    description: Optional[str] = None
    last_known_location: Optional[str] = None
    last_known_time: Optional[datetime] = None
    case_id: int
    status: Optional[str] = "MISSING"

class PersonCreate(PersonBase):
    pass

class PersonUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    description: Optional[str] = None
    last_known_location: Optional[str] = None
    last_known_time: Optional[datetime] = None
    case_id: Optional[int] = None
    status: Optional[str] = None

class PersonResponse(PersonBase):
    id: int
    created_at: datetime
    updated_at: datetime
    photos: List[PhotoResponse] = []
    embeddings: List[EmbeddingResponse] = []

    model_config = ConfigDict(from_attributes=True)
