from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scheduler.cli import _seed_demo_slots
from scheduler.models import Slot
from scheduler.repository import SchedulingRepository


class DemoSeedTests(unittest.TestCase):
    def test_seeds_urgent_care_and_is_safe_to_run_again_on_existing_database(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SchedulingRepository(Path(directory) / "scheduler.db")
            try:
                existing_slot = Slot(
                    id="user-created-slot",
                    specialty="neurology",
                    clinician_name="Dr. Existing",
                    starts_at=datetime.now(timezone.utc) + timedelta(days=20),
                )
                repository.create_slot(existing_slot)

                _seed_demo_slots(repository)
                search_start = datetime.now(timezone.utc)
                search_end = search_start + timedelta(days=4)
                urgent_care_slots = repository.list_available_slots(
                    "Urgent Care",
                    starts_after=search_start,
                    starts_before=search_end,
                )
                count_after_first_seed = len(urgent_care_slots)
                _seed_demo_slots(repository)
                urgent_care_slots_after_second_seed = repository.list_available_slots(
                    "urgent care",
                    starts_after=search_start,
                    starts_before=search_end,
                )

                self.assertGreater(count_after_first_seed, 0)
                self.assertEqual(len(urgent_care_slots_after_second_seed), count_after_first_seed)
                self.assertTrue(repository.has_slot(existing_slot.id))
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
