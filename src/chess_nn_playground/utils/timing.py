from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    info = {"start": time.time(), "elapsed": 0.0}
    try:
        yield info
    finally:
        info["elapsed"] = time.time() - info["start"]
