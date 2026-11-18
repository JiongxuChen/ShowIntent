"""A visible JSONL event journal with an explicit completion/failure sidecar."""

import json
from pathlib import Path


class EventJournal:
    """Create once, append complete flushed lines, never truncate a prior run."""

    def __init__(self, path, *, status_writer):
        self.path = Path(path)
        self.status_path = self.path.with_suffix(".status.json")
        self._write_status = status_writer
        self.stream = None
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.status_path.exists():
            raise ValueError("journal or status already exists; choose a new --journal path")
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        try:
            self._write_status(self.status_path, [{"status": "RUNNING", "event_count": 0,
                                                  "availability": "in_memory_publication_before_transport"}])
        except Exception:
            self.stream.close()
            raise
        return self

    def append(self, events):
        for event in events:
            self.stream.write(json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            self.count += 1
        self.stream.flush()

    def __exit__(self, exc_type, exc, traceback):
        self.stream.close()
        record = {"status": "FAILED" if exc_type else "COMPLETE", "event_count": self.count,
                  "availability": "in_memory_publication_before_transport"}
        if exc_type:
            record["error"] = str(exc)
        self._write_status(self.status_path, [record])
        return False
