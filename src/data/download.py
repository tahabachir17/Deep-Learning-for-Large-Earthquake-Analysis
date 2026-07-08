"""Dataset download placeholders."""

from __future__ import annotations


def get_known_sources() -> dict[str, str]:
    return {
        "synthetic_chile": "https://doi.org/10.5281/zenodo.4008690",
        "real_waveforms": "https://doi.org/10.5281/zenodo.1434374",
    }
