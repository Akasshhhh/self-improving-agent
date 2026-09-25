from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from scheduler.models import Slot
from scheduler.repository import SchedulingRepository
from scheduler.tools import SchedulingTools, ToolName


class SchedulingToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.repository = SchedulingRepository(
            Path(self.temporary_directory.name) / "scheduling.db"
        )
        self.slot = Slot(
            id="cardiology-1",
            specialty="cardiology",
            clinician_name="Dr. Rivera",
            starts_at=datetime(2026, 9, 28, 10, tzinfo=timezone.utc),
        )
        self.repository.create_slot(self.slot)
        self.tools = SchedulingTools(self.repository, patient_id="patient-1")

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary_directory.cleanup()

    def test_search_returns_json_safe_slots(self) -> None:
        result = self.tools.execute(
            ToolName.SEARCH_AVAILABLE_SLOTS, {"specialty": "cardiology"}
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.data["slots"][0]["id"], self.slot.id)
        self.assertEqual(result.data["slots"][0]["starts_at"], "2026-09-28T10:00:00+00:00")

    def test_search_formats_slot_time_in_configured_local_timezone(self) -> None:
        local_tools = SchedulingTools(
            self.repository,
            patient_id="patient-1",
            display_timezone=ZoneInfo("Asia/Kolkata"),
        )

        result = local_tools.execute(
            ToolName.SEARCH_AVAILABLE_SLOTS, {"specialty": "cardiology"}
        )

        slot = result.data["slots"][0]
        self.assertEqual(slot["starts_at"], "2026-09-28T15:30:00+05:30")
        self.assertEqual(slot["timezone"], "Asia/Kolkata")

    def test_invalid_model_arguments_are_rejected_before_booking(self) -> None:
        result = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id, "patient_id": "patient-2"},
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "invalid_arguments")
        self.assertEqual(self.repository.list_available_slots("cardiology"), [self.slot])

    def test_unavailable_slot_returns_a_safe_structured_failure(self) -> None:
        booked = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id},
        )
        self.assertTrue(booked.succeeded)
        self.assertNotIn("patient_id", booked.data["appointment"])
        self.assertEqual(len(self.repository.list_patient_appointments("patient-1")), 1)

        result = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id},
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "slot_unavailable")

    def test_trusted_session_identity_limits_cancellation(self) -> None:
        appointment = self.repository.book_slot("patient-1", self.slot.id)
        other_patient_tools = SchedulingTools(self.repository, patient_id="patient-2")

        result = other_patient_tools.execute(
            ToolName.CANCEL_APPOINTMENT, {"appointment_id": appointment.id}
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "appointment_not_found")

    def test_unknown_tool_is_rejected(self) -> None:
        result = self.tools.execute("delete_everything", {})

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "unknown_tool")


if __name__ == "__main__":
    unittest.main()
