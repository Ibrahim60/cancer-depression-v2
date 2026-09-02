"""Non-interactive reset of submission state. Requires no stdin."""

from submission_tracker import SubmissionState

with SubmissionState() as state:
    state.reset()
print("Tracker reset successfully.")
