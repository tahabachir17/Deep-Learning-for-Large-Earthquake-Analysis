"""Print known dataset download sources."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import get_known_sources


if __name__ == "__main__":
    for name, url in get_known_sources().items():
        print(f"{name}: {url}")
