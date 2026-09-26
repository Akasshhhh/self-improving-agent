"""Bounded, synthetic scenarios proposed from live trace failures."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .evaluation import Scenario
from .models import Slot
from .tools import ToolName


class RelativeSlot(BaseModel):
    """A future slot whose date remains useful on later evaluation runs."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,39}$")
    specialty: str = Field(min_length=2, max_length=60)
    clinician_name: str = Field(min_length=2, max_length=80)
    days_from_now: int = Field(ge=1, le=30)
    hour_utc: int = Field(ge=0, le=23)


class GeneratedScenario(BaseModel):
    """Reviewable test data; it contains no scripted assistant responses."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    description: str = Field(min_length=10, max_length=300)
    slots: list[RelativeSlot] = Field(min_length=0, max_length=6)
    seed_appointment_slot_id: str | None = None
    seed_patient: Literal["self", "other"] | None = None
    patient_messages: list[str] = Field(min_length=1, max_length=4)
    expected_active_slot_ids: list[str] = Field(default_factory=list, max_length=5)
    expected_cancelled_count: int = Field(default=0, ge=0, le=2)
    required_tool_names: list[ToolName] = Field(default_factory=list, max_length=4)
    forbidden_tool_names: list[ToolName] = Field(default_factory=list, max_length=4)
    response_goal: str | None = Field(default=None, max_length=240)
    book_after_patient_message: int | None = Field(default=None, ge=1, le=4)
    book_attempt_after_patient_message: int | None = Field(default=None, ge=1, le=4)
    display_timezone: str = Field(default="UTC", max_length=64)

    @field_validator("display_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Scenario timezone must be an IANA timezone.") from error
        return value

    @model_validator(mode="after")
    def check_setup(self) -> "GeneratedScenario":
        slot_ids = [slot.id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("Generated scenario slot IDs must be unique.")
        if self.seed_patient and not self.seed_appointment_slot_id:
            raise ValueError("A seeded patient needs a seeded appointment slot.")
        if self.seed_appointment_slot_id and self.seed_appointment_slot_id not in slot_ids:
            raise ValueError("Seeded appointment must refer to a scenario slot.")
        if self.seed_appointment_slot_id and not self.seed_patient:
            raise ValueError("A seeded appointment needs its patient type.")
        if not set(self.expected_active_slot_ids).issubset(slot_ids):
            raise ValueError("Expected appointments must refer to scenario slots.")
        if set(self.required_tool_names) & set(self.forbidden_tool_names):
            raise ValueError("A tool cannot be both required and forbidden.")
        if any(not message.strip() or len(message) > 500 for message in self.patient_messages):
            raise ValueError("Patient messages must be nonempty and at most 500 characters.")
        if self.book_after_patient_message and self.book_after_patient_message > len(self.patient_messages):
            raise ValueError("Booking confirmation index exceeds patient message count.")
        if self.book_attempt_after_patient_message and self.book_attempt_after_patient_message > len(self.patient_messages):
            raise ValueError("Booking selection index exceeds patient message count.")
        if (
            self.book_attempt_after_patient_message
            and self.book_after_patient_message
            and self.book_attempt_after_patient_message > self.book_after_patient_message
        ):
            raise ValueError("A booking attempt cannot follow its successful booking threshold.")
        return self

    def to_scenario(self, *, reference_time: datetime | None = None) -> Scenario:
        """Resolve relative dates once; share the result across both policy runs."""
        reference = reference_time or datetime.now(timezone.utc)
        day = reference.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        slots = [
            Slot(
                id=slot.id,
                specialty=slot.specialty,
                clinician_name=slot.clinician_name,
                starts_at=day + timedelta(days=slot.days_from_now, hours=slot.hour_utc),
            )
            for slot in self.slots
        ]
        return Scenario(
            id=self.id,
            description=self.description,
            patient_id="evaluation-patient",
            slots=slots,
            patient_messages=self.patient_messages,
            seed_appointment_slot_id=self.seed_appointment_slot_id,
            seed_appointment_patient_id=(
                "other-evaluation-patient" if self.seed_patient == "other" else None
            ),
            expected_active_slot_ids=self.expected_active_slot_ids,
            expected_cancelled_count=self.expected_cancelled_count,
            required_tool_names=[name.value for name in self.required_tool_names],
            forbidden_tool_names=[name.value for name in self.forbidden_tool_names],
            book_after_patient_message=self.book_after_patient_message,
            book_attempt_after_patient_message=self.book_attempt_after_patient_message,
            display_timezone=self.display_timezone,
        )


def load_generated_scenarios(directory: str | Path) -> list[GeneratedScenario]:
    root = Path(directory)
    scenarios = [
        GeneratedScenario.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    ]
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Generated regression scenario IDs must be unique.")
    return scenarios


def load_protected_scenarios(path: str | Path) -> list[GeneratedScenario]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Protected scenarios must be a JSON list.")
    scenarios = [GeneratedScenario.model_validate(item) for item in raw]
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Protected scenario IDs must be unique.")
    return scenarios


def save_generated_scenario(scenario: GeneratedScenario, directory: str | Path) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{scenario.id}.json"
    if target.exists():
        existing = GeneratedScenario.model_validate_json(target.read_text(encoding="utf-8"))
        if existing != scenario:
            raise ValueError(f"Generated scenario ID '{scenario.id}' already exists.")
        return target
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(scenario.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
