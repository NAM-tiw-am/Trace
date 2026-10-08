from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class CaseBase(BaseModel):
    case_number: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = "ACTIVE"

class CaseCreate(CaseBase):
    pass

class CaseUpdate(BaseModel):
    case_number: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class CaseResponse(CaseBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
