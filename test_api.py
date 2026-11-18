from fastapi.testclient import TestClient

from app.main import app
import app.main as api

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive(monkeypatch):
    def fake_predict(text):
        return {
            "label": "Positive",
            "confidence": 0.91,
            "positive_probability": 0.91,
        }

    monkeypatch.setattr(api, "predict_text", fake_predict)
    response = client.post("/predict", json={"text": "A wonderful and memorable movie"})
    assert response.status_code == 200
    assert response.json()["label"] == "Positive"
    assert 0 <= response.json()["confidence"] <= 1


def test_predict_negative(monkeypatch):
    def fake_predict(text):
        return {
            "label": "Negative",
            "confidence": 0.88,
            "positive_probability": 0.12,
        }

    monkeypatch.setattr(api, "predict_text", fake_predict)
    response = client.post("/predict", json={"text": "A boring and disappointing movie"})
    assert response.status_code == 200
    assert response.json()["label"] == "Negative"


def test_missing_text_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_blank_text_returns_422():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_wrong_method_returns_405():
    response = client.get("/predict")
    assert response.status_code == 405
