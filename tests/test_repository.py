from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from scheduler.models import AppointmentStatus, Slot
from scheduler.repository import SchedulingRepository, SlotUnavailable


class SchedulingRepositoryTests(unittest.TestCase):
    def test_creates_missing_parent_directory_for_database(self) -> None:
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "nested" / "data" / "scheduler.db"

            repository = SchedulingRepository(database_path)
            try:
                self.assertTrue(database_path.exists())
            finally:
                repository.close()

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

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary_directory.cleanup()

    def test_booking_removes_slot_from_available_results(self) -> None:
        appointment = self.repository.book_slot("patient-123", self.slot.id)

        self.assertEqual(appointment.status, AppointmentStatus.BOOKED)
        self.assertEqual(self.repository.list_available_slots("cardiology"), [])

    def test_second_booking_of_same_slot_is_rejected(self) -> None:
        self.repository.book_slot("patient-123", self.slot.id)

        with self.assertRaises(SlotUnavailable):
            self.repository.book_slot("patient-456", self.slot.id)

    def test_cancelling_an_appointment_makes_the_slot_available_again(self) -> None:
        appointment = self.repository.book_slot("patient-123", self.slot.id)

        cancelled = self.repository.cancel_appointment("patient-123", appointment.id)

        self.assertEqual(cancelled.status, AppointmentStatus.CANCELLED)
        self.assertEqual(self.repository.list_available_slots("cardiology"), [self.slot])

    def test_available_slot_search_respects_time_bounds(self) -> None:
        later_slot = Slot(
            id="cardiology-2",
            specialty="cardiology",
            clinician_name="Dr. Rivera",
            starts_at=self.slot.starts_at + timedelta(days=7),
        )
        self.repository.create_slot(later_slot)

        result = self.repository.list_available_slots(
            "cardiology", starts_before=self.slot.starts_at + timedelta(days=1)
        )

        self.assertEqual(result, [self.slot])

    def test_available_slot_search_is_case_insensitive_for_specialty(self) -> None:
        result = self.repository.list_available_slots("CARDIOLOGY")

        self.assertEqual(result, [self.slot])

    def test_available_slot_search_converts_local_time_bounds_before_comparing(self) -> None:
        india_time = ZoneInfo("Asia/Kolkata")
        result = self.repository.list_available_slots(
            "cardiology",
            starts_after=datetime(2026, 9, 28, 15, 15, tzinfo=india_time),
            starts_before=datetime(2026, 9, 28, 16, 0, tzinfo=india_time),
        )

        self.assertEqual(result, [self.slot])


if __name__ == "__main__":
    unittest.main()
