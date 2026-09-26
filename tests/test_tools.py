from datetime import datetime, timedelta, timezone
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

    def test_list_appointments_returns_visit_details_and_upcoming_status(self) -> None:
        future_slot = Slot(
            id="future-visit",
            specialty="urgent care",
            clinician_name="Urgent Care Team",
            starts_at=datetime.now(timezone.utc) + timedelta(days=2),
        )
        past_slot = Slot(
            id="past-visit",
            specialty="dermatology",
            clinician_name="Dr. Chen",
            starts_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        cancelled_slot = Slot(
            id="cancelled-visit",
            specialty="primary care",
            clinician_name="Dr. Patel",
            starts_at=datetime.now(timezone.utc) + timedelta(days=3),
        )
        self.repository.create_slot(future_slot)
        self.repository.create_slot(past_slot)
        self.repository.create_slot(cancelled_slot)
        self.repository.book_slot("patient-1", future_slot.id)
        self.repository.book_slot("patient-1", past_slot.id)
        cancelled = self.repository.book_slot("patient-1", cancelled_slot.id)
        self.repository.cancel_appointment("patient-1", cancelled.id)
        self.repository.book_slot("other-patient", self.slot.id)
        local_tools = SchedulingTools(
            self.repository,
            patient_id="patient-1",
            display_timezone=ZoneInfo("Asia/Kolkata"),
        )

        result = local_tools.execute(ToolName.LIST_MY_APPOINTMENTS, {})

        self.assertTrue(result.succeeded)
        appointments = {item["slot_id"]: item for item in result.data["appointments"]}
        self.assertEqual(set(appointments), {"future-visit", "past-visit", "cancelled-visit"})
        self.assertTrue(appointments["future-visit"]["is_upcoming"])
        self.assertFalse(appointments["past-visit"]["is_upcoming"])
        self.assertFalse(appointments["cancelled-visit"]["is_upcoming"])
        self.assertEqual(appointments["future-visit"]["slot"]["specialty"], "urgent care")
        self.assertEqual(appointments["future-visit"]["slot"]["timezone"], "Asia/Kolkata")
        self.assertNotIn("created_at", appointments["future-visit"])
        self.assertNotIn("patient_id", appointments["future-visit"])

    def test_invalid_model_arguments_are_rejected_before_booking(self) -> None:
        result = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id, "patient_id": "patient-2"},
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "invalid_arguments")
        self.assertEqual(self.repository.list_available_slots("cardiology"), [self.slot])

    def test_unavailable_slot_returns_a_safe_structured_failure(self) -> None:
        unconfirmed = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id},
        )
        self.assertFalse(unconfirmed.succeeded)
        self.assertEqual(unconfirmed.error_code, "confirmation_required")
        self.assertEqual(self.repository.list_patient_appointments("patient-1"), [])

        booked = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id},
            confirmed_booking_slot_id=self.slot.id,
        )
        self.assertTrue(booked.succeeded)
        self.assertNotIn("patient_id", booked.data["appointment"])
        self.assertEqual(len(self.repository.list_patient_appointments("patient-1")), 1)

        result = self.tools.execute(
            ToolName.BOOK_APPOINTMENT,
            {"slot_id": self.slot.id},
            confirmed_booking_slot_id=self.slot.id,
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
