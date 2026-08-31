"""
Submission Tracker
==================
Tracks which records have been submitted to avoid duplicates.
Creates and manages a submission log file.
"""

import json
import os
from datetime import datetime
from typing import Set, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(SCRIPT_DIR, "submission_log.json")


class SubmissionTracker:
    """Manages tracking of submitted records to prevent duplicates."""
    
    def __init__(self, tracker_file: str = TRACKER_FILE):
        self.tracker_file = tracker_file
        self.data = self._load_tracker()
    
    def _load_tracker(self) -> Dict:
        """Load existing tracker data or create new structure."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Initialize new tracker structure
        return {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_submitted": 0,
            "submitted_records": {},  # {record_index: {name, timestamp, status}}
            "failed_records": {},     # {record_index: {name, timestamp, error}}
            "session_history": []     # List of past sessions
        }
    
    def _save_tracker(self):
        """Save tracker data to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def is_submitted(self, record_index: int) -> bool:
        """Check if a record has already been submitted."""
        return str(record_index) in self.data["submitted_records"]
    
    def is_failed(self, record_index: int) -> bool:
        """Check if a record previously failed."""
        return str(record_index) in self.data["failed_records"]
    
    def mark_submitted(self, record_index: int, name: str, fields_filled: int = 0):
        """Mark a record as successfully submitted."""
        record_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "fields_filled": fields_filled,
            "status": "submitted"
        }
        self.data["submitted_records"][str(record_index)] = record_data
        self.data["total_submitted"] = len(self.data["submitted_records"])
        self._save_tracker()
    
    def mark_failed(self, record_index: int, name: str, error: str):
        """Mark a record as failed."""
        record_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "status": "failed"
        }
        self.data["failed_records"][str(record_index)] = record_data
        self._save_tracker()
    
    def mark_retry(self, record_index: int):
        """Remove a failed record so it can be retried."""
        if str(record_index) in self.data["failed_records"]:
            del self.data["failed_records"][str(record_index)]
            self._save_tracker()
    
    def get_pending_indices(self, total_records: int) -> List[int]:
        """Get list of record indices that haven't been submitted yet."""
        submitted = set(int(idx) for idx in self.data["submitted_records"].keys())
        failed = set(int(idx) for idx in self.data["failed_records"].keys())
        done = submitted | failed
        return [i for i in range(total_records) if i not in done]
    
    def get_submitted_count(self) -> int:
        """Get total number of successfully submitted records."""
        return len(self.data["submitted_records"])
    
    def get_failed_count(self) -> int:
        """Get total number of failed records."""
        return len(self.data["failed_records"])
    
    def get_summary(self) -> str:
        """Get a summary of submission status."""
        return (
            f"Submission Summary:\n"
            f"  Total Submitted: {self.get_submitted_count()}\n"
            f"  Failed: {self.get_failed_count()}\n"
            f"  Last Updated: {self.data['last_updated']}"
        )
    
    def start_session(self, session_name: str = ""):
        """Start a new submission session."""
        session_data = {
            "name": session_name or f"Session {len(self.data['session_history']) + 1}",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "records_submitted": 0
        }
        self.data["session_history"].append(session_data)
        self._save_tracker()
        return len(self.data["session_history"]) - 1
    
    def end_session(self, session_index: int, records_submitted: int):
        """End a submission session."""
        if 0 <= session_index < len(self.data["session_history"]):
            self.data["session_history"][session_index]["end_time"] = datetime.now().isoformat()
            self.data["session_history"][session_index]["records_submitted"] = records_submitted
            self._save_tracker()
    
    def reset_tracker(self):
        """Reset the tracker (use with caution)."""
        self.data = self._load_tracker()
        self.data["submitted_records"] = {}
        self.data["failed_records"] = {}
        self.data["total_submitted"] = 0
        self.data["last_updated"] = datetime.now().isoformat()
        self._save_tracker()
    
    def export_report(self, output_file: Optional[str] = None):
        """Export a detailed report of submissions."""
        if output_file is None:
            output_file = os.path.join(SCRIPT_DIR, "submission_report.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SUBMISSION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Created: {self.data['created']}\n")
            f.write(f"Last Updated: {self.data['last_updated']}\n\n")
            f.write(f"Total Submitted: {self.get_submitted_count()}\n")
            f.write(f"Total Failed: {self.get_failed_count()}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("SUBMITTED RECORDS\n")
            f.write("-" * 60 + "\n")
            for idx, record in sorted(self.data["submitted_records"].items(), key=lambda x: int(x[0])):
                f.write(f"Record {idx}: {record['name']} - {record['timestamp']}\n")
                f.write(f"  Fields filled: {record.get('fields_filled', 'N/A')}\n")
            
            if self.data["failed_records"]:
                f.write("\n" + "-" * 60 + "\n")
                f.write("FAILED RECORDS\n")
                f.write("-" * 60 + "\n")
                for idx, record in sorted(self.data["failed_records"].items(), key=lambda x: int(x[0])):
                    f.write(f"Record {idx}: {record['name']} - {record['timestamp']}\n")
                    f.write(f"  Error: {record['error']}\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("SESSION HISTORY\n")
            f.write("-" * 60 + "\n")
            for i, session in enumerate(self.data["session_history"], 1):
                f.write(f"Session {i}: {session['name']}\n")
                f.write(f"  Start: {session['start_time']}\n")
                f.write(f"  End: {session.get('end_time', 'In progress')}\n")
                f.write(f"  Records submitted: {session['records_submitted']}\n")
        
        return output_file


# Command-line interface for tracker management
def main():
    import sys
    
    tracker = SubmissionTracker()
    
    if len(sys.argv) < 2:
        print("Usage: python submission_tracker.py <command>")
        print("Commands:")
        print("  status  - Show submission status")
        print("  summary - Show summary")
        print("  report  - Export detailed report")
        print("  reset   - Reset tracker (WARNING: deletes all records)")
        print("  pending <total> - Show pending record indices")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        print(tracker.get_summary())
        print(f"\nSubmitted Records: {tracker.get_submitted_count()}")
        print(f"Failed Records: {tracker.get_failed_count()}")
        
    elif command == "summary":
        print(tracker.get_summary())
        
    elif command == "report":
        report_file = tracker.export_report()
        print(f"Report exported to: {report_file}")
        
    elif command == "reset":
        confirm = input("Are you sure you want to reset the tracker? This will delete all submission records. (yes/no): ")
        if confirm.lower() == "yes":
            tracker.reset_tracker()
            print("Tracker reset successfully.")
        else:
            print("Reset cancelled.")
            
    elif command == "pending":
        if len(sys.argv) < 3:
            print("Usage: python submission_tracker.py pending <total_records>")
            return
        total = int(sys.argv[2])
        pending = tracker.get_pending_indices(total)
        print(f"Pending records: {len(pending)} out of {total}")
        if pending:
            print(f"Pending indices: {pending[:20]}{'...' if len(pending) > 20 else ''}")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
