"""SQLite-backed scheduling operations that own all appointment side effects."""

import sqlite3
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from uuid import uuid4

from .models import Appointment, AppointmentStatus, Slot


class SchedulingError(Exception):
    """Base class for expected scheduling failures."""


class SlotNotFound(SchedulingError):
    """Raised when a requested slot does not exist."""


class SlotUnavailable(SchedulingError):
    """Raised when another active appointment already owns a slot."""


class AppointmentNotFound(SchedulingError):
    """Raised when a patient cannot cancel the requested appointment."""


class SchedulingRepository:
    """The authoritative persistence boundary for slots and appointments."""

    def __init__(
        self,
        database_path: str | Path,
        appointment_id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self._appointment_id_factory = appointment_id_factory
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def has_slots(self) -> bool:
        return self._connection.execute("SELECT 1 FROM slots LIMIT 1").fetchone() is not None

    def list_patient_appointments(self, patient_id: str) -> list[Appointment]:
        rows = self._connection.execute(
            """
            SELECT id, patient_id, slot_id, status, created_at
            FROM appointments WHERE patient_id = ? ORDER BY created_at
            """,
            (patient_id,),
        ).fetchall()
        return [
            Appointment(
                id=row["id"],
                patient_id=row["patient_id"],
                slot_id=row["slot_id"],
                status=AppointmentStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def create_slot(self, slot: Slot) -> Slot:
        self._connection.execute(
            """
            INSERT INTO slots (id, specialty, clinician_name, starts_at)
            VALUES (?, ?, ?, ?)
            """,
            (slot.id, slot.specialty, slot.clinician_name, slot.starts_at.isoformat()),
        )
        self._connection.commit()
        return slot

    def list_available_slots(
        self,
        specialty: str,
        starts_after: datetime | None = None,
        starts_before: datetime | None = None,
    ) -> list[Slot]:
        clauses = ["s.specialty = ?", "a.id IS NULL"]
        values: list[str] = [specialty]
        if starts_after is not None:
            clauses.append("s.starts_at >= ?")
            values.append(starts_after.isoformat())
        if starts_before is not None:
            clauses.append("s.starts_at < ?")
            values.append(starts_before.isoformat())

        rows = self._connection.execute(
            f"""
            SELECT s.id, s.specialty, s.clinician_name, s.starts_at
            FROM slots AS s
            LEFT JOIN appointments AS a
                ON a.slot_id = s.id AND a.status = '{AppointmentStatus.BOOKED.value}'
            WHERE {' AND '.join(clauses)}
            ORDER BY s.starts_at
            """,
            values,
        ).fetchall()
        return [self._slot_from_row(row) for row in rows]

    def book_slot(self, patient_id: str, slot_id: str) -> Appointment:
        """Book one slot atomically or raise a domain error without side effects."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            slot_exists = self._connection.execute(
                "SELECT 1 FROM slots WHERE id = ?", (slot_id,)
            ).fetchone()
            if slot_exists is None:
                raise SlotNotFound(f"Slot '{slot_id}' does not exist.")

            booked = self._connection.execute(
                """
                SELECT 1 FROM appointments
                WHERE slot_id = ? AND status = ?
                """,
                (slot_id, AppointmentStatus.BOOKED.value),
            ).fetchone()
            if booked is not None:
                raise SlotUnavailable(f"Slot '{slot_id}' is no longer available.")

            appointment = Appointment(
                id=self._appointment_id_factory(),
                patient_id=patient_id,
                slot_id=slot_id,
                status=AppointmentStatus.BOOKED,
                created_at=datetime.now().astimezone(),
            )
            self._connection.execute(
                """
                INSERT INTO appointments (id, patient_id, slot_id, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    appointment.id,
                    appointment.patient_id,
                    appointment.slot_id,
                    appointment.status.value,
                    appointment.created_at.isoformat(),
                ),
            )
            self._connection.commit()
            return appointment
        except Exception:
            self._connection.rollback()
            raise

    def cancel_appointment(self, patient_id: str, appointment_id: str) -> Appointment:
        row = self._connection.execute(
            """
            SELECT id, patient_id, slot_id, status, created_at
            FROM appointments
            WHERE id = ? AND patient_id = ? AND status = ?
            """,
            (appointment_id, patient_id, AppointmentStatus.BOOKED.value),
        ).fetchone()
        if row is None:
            raise AppointmentNotFound("No active appointment was found for this patient.")

        self._connection.execute(
            "UPDATE appointments SET status = ? WHERE id = ?",
            (AppointmentStatus.CANCELLED.value, appointment_id),
        )
        self._connection.commit()
        return Appointment(
            id=row["id"],
            patient_id=row["patient_id"],
            slot_id=row["slot_id"],
            status=AppointmentStatus.CANCELLED,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS slots (
                id TEXT PRIMARY KEY,
                specialty TEXT NOT NULL,
                clinician_name TEXT NOT NULL,
                starts_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                slot_id TEXT NOT NULL REFERENCES slots(id),
                status TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')),
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS one_active_booking_per_slot
                ON appointments(slot_id)
                WHERE status = 'booked';
            """
        )
        self._connection.commit()

    @staticmethod
    def _slot_from_row(row: sqlite3.Row) -> Slot:
        return Slot(
            id=row["id"],
            specialty=row["specialty"],
            clinician_name=row["clinician_name"],
            starts_at=datetime.fromisoformat(row["starts_at"]),
        )
