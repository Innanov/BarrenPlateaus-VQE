"""Progress and timing helpers: a duration formatter and an elapsed / ETA logger."""

import time


def format_duration(seconds: float) -> str:
    """Format a duration as MM:SS (or H:MM:SS past an hour)."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class Progress:
    """Elapsed / ETA logger that prefixes every line with a running clock.

    step marks one completed work-unit and prints a message with an
    [elapsed | ETA | done/total] prefix (ETA extrapolated from the mean time per
    unit). note prints with the same prefix without advancing.

    Attributes:
        total: Total number of work-units expected.
        start: Wall-clock start time (time.monotonic).
        done: Work-units completed so far.
    """

    def __init__(self, total: int):
        """Initialize the progress clock for total expected work-units."""
        self.total = max(1, int(total))
        self.start = time.monotonic()
        self.done = 0

    def _prefix(self) -> str:
        """Build the [elapsed | ETA rem | done/total] prefix from progress so far."""
        elapsed = time.monotonic() - self.start
        if self.done <= 0:
            eta = "?"
        else:
            per = elapsed / self.done
            eta = format_duration(per * (self.total - self.done))
        return f"[{format_duration(elapsed)} elapsed | ETA {eta} | {self.done}/{self.total}]"

    def step(self, message: str) -> None:
        """Advance one work-unit and print message with the clock prefix."""
        self.done += 1
        print(f"{self._prefix()} {message}", flush=True)

    def note(self, message: str) -> None:
        """Print message with the clock prefix without advancing progress."""
        print(f"{self._prefix()} {message}", flush=True)
