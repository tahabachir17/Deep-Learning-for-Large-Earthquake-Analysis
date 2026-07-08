"""Optional API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    waveform: list
    nst: int = Field(..., ge=1)
    nt: int = Field(..., ge=1)
    model_path: str


class PredictResponse(BaseModel):
    predictions: list[float]
