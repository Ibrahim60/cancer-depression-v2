"""
Crash-safe, fingerprint-keyed submission state.

State is written atomically (temp file + fsync + os.replace) and guarded by
an exclusive file lock so two submit.py processes cannot run at once.

Cross-platform locking: uses fcntl on Unix and msvcrt.locking on Windows.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

try:
    import msvcrt
except ImportError:  # pragma: no cover - Unix
    msvcrt = None  # type: ignore


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE_FILE = os.path.join(SCRIPT_DIR, "submission_state.json")
DEFAULT_LOCK_FILE = os.path.join(SCRIPT_DIR, "submit.lock")
STATE_VERSION = 2

VALID_STATUSES = ("pending", "success", "failed", "skipped", "unknown")


class StateError(Exception):
    """Unrecoverable state-file problem."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created": _utc_now(),
        "last_updated": _utc_now(),
        "records": {},
        "sessions": [],
        "legacy_index_state": None,
    }


def _is_legacy_format(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and "submitted_records" in data
        and "records" not in data
    )


class SubmissionState:
    """Persistent per-record submission ledger."""

    def __init__(
        self,
        state_file: str = DEFAULT_STATE_FILE,
        lock_file: str = DEFAULT_LOCK_FILE,
    ):
        self.state_file = state_file
        self.bak_file = state_file + ".bak"
        self.lock_file = lock_file
        self._lock_fh = None
        self.data: Dict[str, Any] = _empty_state()

    def __enter__(self) -> "SubmissionState":
        self.acquire_lock()
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.save()
        finally:
            self.release_lock()

    # ── locking ──────────────────────────────────────────────────────────────

    def acquire_lock(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_file)) or ".", exist_ok=True)
        self._lock_fh = open(self.lock_file, "a+", encoding="utf-8")
        
        if fcntl is not None:
            # Unix: use fcntl
            try:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock_fh.close()
                self._lock_fh = None
                raise StateError(
                    f"Another submit.py instance holds {self.lock_file}. "
                    "Only one process may run at a time."
                ) from exc
        elif msvcrt is not None:
            # Windows: use msvcrt.locking
            try:
                # For Windows, we need to lock a specific region of the file
                # Lock the first 100 bytes of the file
                self._lock_fh.seek(0)
                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_NBLCK, 100)
            except OSError as exc:
                self._lock_fh.close()
                self._lock_fh = None
                raise StateError(
                    f"Another submit.py instance holds {self.lock_file}. "
                    "Only one process may run at a time."
                ) from exc
        else:
            raise StateError(
                "Neither fcntl nor msvcrt is available; refusing to run without an exclusive lock."
            )
        
        # Write PID after acquiring lock (common for both platforms)
        self._lock_fh.seek(0)
        self._lock_fh.truncate()
        self._lock_fh.write(str(os.getpid()))
        self._lock_fh.flush()

    def release_lock(self) -> None:
        if self._lock_fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                # Windows: unlock the file
                self._lock_fh.seek(0)
                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 100)
        finally:
            self._lock_fh.close()
            self._lock_fh = None

    # ── load / save ──────────────────────────────────────────────────────────

    def load(self) -> None:
        if not os.path.exists(self.state_file):
            if os.path.exists(self.bak_file):
                print(
                    f"STATE primary missing; recovering from backup {self.bak_file}",
                    file=sys.stderr,
                )
                self.data = self._read_json_file(self.bak_file)
                self.save()
                return
            self.data = _empty_state()
            self.save()
            return

        try:
            self.data = self._read_json_file(self.state_file)
        except StateError:
            corrupt_path = (
                f"{self.state_file}.corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            try:
                shutil.copy2(self.state_file, corrupt_path)
                print(f"STATE preserved damaged file as {corrupt_path}", file=sys.stderr)
            except OSError as copy_exc:
                print(f"STATE could not preserve damaged file: {copy_exc}", file=sys.stderr)

            if os.path.exists(self.bak_file):
                print(f"STATE recovering from backup {self.bak_file}", file=sys.stderr)
                self.data = self._read_json_file(self.bak_file)
                self.save()
                return

            raise StateError(
                f"State file is unreadable and no backup exists: {self.state_file}. "
                "Refusing to start with an empty ledger (would risk duplicate submissions). "
                f"Damaged file kept at {corrupt_path}."
            )

        if _is_legacy_format(self.data):
            self._adopt_unmapped_legacy()

        self.data.setdefault("version", STATE_VERSION)
        self.data.setdefault("records", {})
        self.data.setdefault("sessions", [])
        self.data.setdefault("created", _utc_now())

    def _read_json_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(f"Cannot parse state file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise StateError(f"State file {path} is not a JSON object.")
        return data

    def _adopt_unmapped_legacy(self) -> None:
        """Keep old index-based history; do not treat it as fingerprint success."""
        legacy_path = self.state_file + ".legacy"
        try:
            shutil.copy2(self.state_file, legacy_path)
        except OSError:
            pass
        print(
            "STATE detected legacy index-based tracker. "
            f"Copied to {legacy_path}. "
            "Index identities cannot be mapped to fingerprints safely, so those "
            "rows are NOT assumed successful. Pass --adopt-legacy-by-index when "
            "launching submit.py only if the CSV order is unchanged.",
            file=sys.stderr,
        )
        preserved = self.data
        self.data = _empty_state()
        self.data["legacy_index_state"] = preserved

    def adopt_legacy_by_index(self, fingerprints_in_order: List[str], names: List[str]) -> int:
        """Map legacy submitted_records[i] → fingerprints[i]. Unsafe if CSV reordered."""
        legacy = self.data.get("legacy_index_state") or {}
        submitted = legacy.get("submitted_records") or {}
        mapped = 0
        for idx_str, rec in submitted.items():
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(fingerprints_in_order):
                rid = fingerprints_in_order[idx]
                name = names[idx] if idx < len(names) else (rec.get("name") or "")
                entry = self.ensure_record(rid, name=name, source_row=idx)
                if entry.get("status") != "success":
                    self.update_record(
                        rid,
                        status="success",
                        name=name,
                        source_row=idx,
                        success_at=rec.get("timestamp") or _utc_now(),
                        fields_filled=rec.get("fields_filled") or 0,
                        error_type=None,
                        error_message="adopted from legacy index-based tracker",
                    )
                    mapped += 1
        self.data["legacy_index_state"] = None
        self.save()
        return mapped

    def save(self) -> None:
        self.data["last_updated"] = _utc_now()
        self.data["version"] = STATE_VERSION
        directory = os.path.dirname(os.path.abspath(self.state_file)) or "."
        os.makedirs(directory, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix=".state_", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            if os.path.exists(self.state_file):
                shutil.copy2(self.state_file, self.bak_file)
            os.replace(tmp_path, self.state_file)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── record API ───────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        rec = self.data["records"].get(record_id)
        return rec

    def ensure_record(self, record_id: str, **fields: Any) -> Dict[str, Any]:
        records = self.data["records"]
        if record_id not in records:
            records[record_id] = {
                "record_id": record_id,
                "status": "pending",
                "name": fields.get("name") or "",
                "attempt_count": 0,
                "first_attempt_at": None,
                "last_attempt_at": None,
                "success_at": None,
                "error_type": None,
                "error_message": None,
                "failure_class": None,
                "fields_filled": 0,
                "confirmation": None,
                "source_row": fields.get("source_row"),
                "skip_reason": None,
            }
            self.save()
        return records[record_id]

    def update_record(self, record_id: str, **fields: Any) -> Dict[str, Any]:
        rec = self.ensure_record(record_id)
        for key, value in fields.items():
            rec[key] = value
        rec["record_id"] = record_id
        if rec.get("status") not in VALID_STATUSES:
            raise StateError(f"Invalid status {rec.get('status')!r} for {record_id}")
        self.save()
        return rec

    def counts(self) -> Dict[str, int]:
        tallies = {s: 0 for s in VALID_STATUSES}
        for rec in self.data["records"].values():
            status = rec.get("status", "pending")
            tallies[status] = tallies.get(status, 0) + 1
        return tallies

    def start_session(self) -> int:
        session = {
            "start_time": _utc_now(),
            "end_time": None,
            "pid": os.getpid(),
        }
        self.data["sessions"].append(session)
        self.save()
        return len(self.data["sessions"]) - 1

    def end_session(self, index: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if 0 <= index < len(self.data["sessions"]):
            self.data["sessions"][index]["end_time"] = _utc_now()
            if extra:
                self.data["sessions"][index].update(extra)
            self.save()

    def reset(self) -> None:
        created = self.data.get("created") or _utc_now()
        self.data = _empty_state()
        self.data["created"] = created
        self.save()

    def export_report(self, output_file: Optional[str] = None) -> str:
        if output_file is None:
            output_file = os.path.join(
                os.path.dirname(os.path.abspath(self.state_file)),
                "submission_report.txt",
            )
        counts = self.counts()
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write("SUBMISSION REPORT\n")
            fh.write(f"Created: {self.data.get('created')}\n")
            fh.write(f"Last Updated: {self.data.get('last_updated')}\n")
            fh.write(f"Counts: {counts}\n\n")
            for rid, rec in sorted(self.data["records"].items()):
                fh.write(
                    f"{rid[:12]}…  {rec.get('status'):<9}  "
                    f"{rec.get('name', '')}  attempts={rec.get('attempt_count', 0)}\n"
                )
                if rec.get("error_message"):
                    fh.write(f"    error: {rec['error_message']}\n")
        return output_file


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python submission_tracker.py <status|report|reset --confirm>")
        return

    command = args[0].lower()
    with SubmissionState() as state:
        if command == "status":
            print(json.dumps({"counts": state.counts(), "updated": state.data.get("last_updated")}, indent=2))
        elif command == "report":
            path = state.export_report()
            print(f"Report exported to: {path}")
        elif command == "reset":
            if "--confirm" not in args:
                print("Refusing to reset without --confirm (non-interactive).")
                sys.exit(2)
            state.reset()
            print("Tracker reset successfully.")
        else:
            print(f"Unknown command: {command}")
            sys.exit(2)


if __name__ == "__main__":
    main()
