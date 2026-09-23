"""Typed domain records shared by tools, the repository, and evaluation."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AppointmentStatus(StrEnum):
    BOOKED = "booked"
    CANCELLED = "cancelled"


class Slot(BaseModel):
    """A time a clinician can offer for one specialty."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    specialty: str = Field(min_length=1)
    clinician_name: str = Field(min_length=1)
    starts_at: datetime


class Appointment(BaseModel):
    """The durable record created when a patient books a slot."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    status: AppointmentStatus
    created_at: datetime

