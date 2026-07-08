"""FastAPI app for model inference."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI

from src.inference.predict import load_and_predict
from src.inference.schema import PredictRequest, PredictResponse

app = FastAPI(title="earthquake-dl-hrgnss")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    waveform = np.asarray(request.waveform, dtype=np.float32)
    predictions = load_and_predict(request.model_path, waveform, nst=request.nst, nt=request.nt, round_digits=1)
    return PredictResponse(predictions=[float(value) for value in predictions])
