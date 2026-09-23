"""Validated, allow-listed tools exposed to the scheduling agent."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import Appointment, Slot
from .repository import AppointmentNotFound, SchedulingError, SchedulingRepository, SlotNotFound, SlotUnavailable


class ToolName(StrEnum):
    SEARCH_AVAILABLE_SLOTS = "search_available_slots"
    LIST_MY_APPOINTMENTS = "list_my_appointments"
    BOOK_APPOINTMENT = "book_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"


class SearchAvailableSlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specialty: str = Field(min_length=1)
    starts_after: datetime | None = None
    starts_before: datetime | None = None


class BookAppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1)


class CancelAppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appointment_id: str = Field(min_length=1)


class ListMyAppointmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    """A JSON-safe result that can be returned to the model as tool output."""

    tool_name: str
    succeeded: bool
    data: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


class SchedulingTools:
    """Converts untrusted tool arguments into validated repository calls."""

    def __init__(self, repository: SchedulingRepository, patient_id: str) -> None:
        self._repository = repository
        if not patient_id.strip():
            raise ValueError("A trusted patient identity is required.")
        self._patient_id = patient_id

    def execute(self, tool_name: ToolName | str, arguments: dict[str, Any]) -> ToolResult:
        """Run one allow-listed tool and convert expected errors into safe output."""
        try:
            name = ToolName(tool_name)
            if name is ToolName.SEARCH_AVAILABLE_SLOTS:
                request = SearchAvailableSlotsInput.model_validate(arguments)
                slots = self._repository.list_available_slots(
                    specialty=request.specialty,
                    starts_after=request.starts_after,
                    starts_before=request.starts_before,
                )
                return self._success(name, {"slots": [self._serialize(slot) for slot in slots]})

            if name is ToolName.BOOK_APPOINTMENT:
                request = BookAppointmentInput.model_validate(arguments)
                appointment = self._repository.book_slot(self._patient_id, request.slot_id)
                return self._success(name, {"appointment": self._serialize(appointment)})

            if name is ToolName.LIST_MY_APPOINTMENTS:
                ListMyAppointmentsInput.model_validate(arguments)
                appointments = self._repository.list_patient_appointments(self._patient_id)
                return self._success(
                    name,
                    {"appointments": [self._serialize(item) for item in appointments]},
                )

            request = CancelAppointmentInput.model_validate(arguments)
            appointment = self._repository.cancel_appointment(
                self._patient_id, request.appointment_id
            )
            return self._success(name, {"appointment": self._serialize(appointment)})
        except ValidationError as error:
            return ToolResult(
                tool_name=str(tool_name),
                succeeded=False,
                error_code="invalid_arguments",
                error_message=error.errors()[0]["msg"],
            )
        except SlotNotFound as error:
            return self._failure(tool_name, "slot_not_found", str(error))
        except SlotUnavailable as error:
            return self._failure(tool_name, "slot_unavailable", str(error))
        except AppointmentNotFound as error:
            return self._failure(tool_name, "appointment_not_found", str(error))
        except ValueError:
            return ToolResult(
                tool_name=str(tool_name),
                succeeded=False,
                error_code="unknown_tool",
                error_message="The requested tool is not available.",
            )
        except SchedulingError as error:
            return self._failure(tool_name, "scheduling_error", str(error))

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        """Return schemas suitable for registering these tools with an LLM provider."""
        return [
            {
                "type": "function",
                "function": {
                    "name": ToolName.LIST_MY_APPOINTMENTS.value,
                    "description": "List this patient's appointments.",
                    "parameters": ListMyAppointmentsInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": ToolName.SEARCH_AVAILABLE_SLOTS.value,
                    "description": "Find available appointment slots by specialty and time range.",
                    "parameters": SearchAvailableSlotsInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": ToolName.BOOK_APPOINTMENT.value,
                    "description": "Book one available slot selected by the patient.",
                    "parameters": BookAppointmentInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": ToolName.CANCEL_APPOINTMENT.value,
                    "description": "Cancel an active appointment belonging to this patient.",
                    "parameters": CancelAppointmentInput.model_json_schema(),
                },
            },
        ]

    @staticmethod
    def _serialize(model: Slot | Appointment) -> dict[str, Any]:
        serialized = model.model_dump(mode="json")
        if isinstance(model, Appointment):
            serialized.pop("patient_id", None)
        return serialized

    def _success(self, tool_name: ToolName, data: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_name=tool_name, succeeded=True, data=data)

    def _failure(self, tool_name: ToolName | str, code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_name=str(tool_name),
            succeeded=False,
            error_code=code,
            error_message=message,
        )
