"""Shared raw stdout/stderr recorder for Python scripts."""

from __future__ import annotations

import atexit
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class _Tee:
    def __init__(self, original: TextIO, record: TextIO) -> None:
        self.original = original
        self.record = record

    def write(self, text: str) -> int:
        self.original.write(text)
        self.record.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.record.flush()

    def isatty(self) -> bool:
        return self.original.isatty()


def start_recording(record_name: str, record_dir: Path) -> Path:
    """Tee this process's stdout/stderr to a timestamped raw log."""
    started = _timestamp()
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{started}_{record_name}.log"
    handle = path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, handle)
    sys.stderr = _Tee(original_stderr, handle)

    print(
        "\n".join(
            [
                f"record_name={record_name}",
                f"started_utc={started}",
                f"host={socket.gethostname()}",
                f"user={os.environ.get('USER', 'unknown')}",
                f"working_directory={Path.cwd()}",
                f"command={' '.join(sys.argv)}",
                "--- output ---",
            ]
        )
    )

    def finish() -> None:
        print(
            "\n".join(
                [
                    "--- end ---",
                    f"finished_utc={_timestamp()}",
                    f"raw_record={path}",
                ]
            )
        )
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handle.close()

    atexit.register(finish)
    return path
