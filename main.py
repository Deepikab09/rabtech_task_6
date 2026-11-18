import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VOCAB_SIZE = 10_000
MAX_LENGTH = 250
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/imdb_sentiment_classifier.keras"))

app = FastAPI(
    title="Movie Review Sentiment API",
    description="Real-time Positive/Negative sentiment inference using the Task 05 TensorFlow model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Movie review text")


class PredictionResponse(BaseModel):
    label: Literal["Positive", "Negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    positive_probability: float = Field(..., ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained Task 05 Keras model and the matching IMDB word index once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run Task 05 and copy "
            "imdb_sentiment_classifier.keras into the models/ directory."
        )

    # Lazy imports keep API schema/tests lightweight and avoid loading TensorFlow at import time.
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    word_index = imdb.get_word_index()
    return model, word_index


def encode_review(text: str, word_index: dict) -> np.ndarray:
    tokens = re.findall(r"[a-z']+", text.lower())
    encoded = [1]  # IMDB start token, matching Task 05 preprocessing.
    for token in tokens:
        index = word_index.get(token)
        encoded.append(index + 3 if index is not None and index + 3 < VOCAB_SIZE else 2)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(
        [encoded], maxlen=MAX_LENGTH, padding="post", truncating="post"
    )


def predict_text(text: str) -> PredictionResponse:
    try:
        model, word_index = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model artifacts could not be loaded: {exc}") from exc

    sequence = encode_review(text, word_index)
    try:
        positive_probability = float(model.predict(sequence, verbose=0)[0][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    positive_probability = min(1.0, max(0.0, positive_probability))
    if positive_probability >= 0.5:
        label = "Positive"
        confidence = positive_probability
    else:
        label = "Negative"
        confidence = 1.0 - positive_probability

    return PredictionResponse(
        label=label,
        confidence=round(confidence, 6),
        positive_probability=round(positive_probability, 6),
    )


@app.get("/")
def root():
    return {"message": "Movie Review Sentiment API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file_present": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    return predict_text(request.text)
