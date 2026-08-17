import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from src.api.main import app

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.jpg")


def test_health_endpoint():
    """Verifie que l'endpoint /health repond correctement"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == True


def test_predict_with_valid_image():
    """Verifie que /predict renvoie une prediction correcte pour une vraie image"""
    with TestClient(app) as client:
        with open(TEST_IMAGE_PATH, "rb") as f:
            response = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert isinstance(data["predicted_class"], str)
    assert 0 <= data["confidence"] <= 100


def test_predict_rejects_non_image_file():
    """Verifie que /predict rejette un fichier qui n'est pas une image"""
    with TestClient(app) as client:
        fake_file = ("test.txt", b"ceci n'est pas une image", "text/plain")
        response = client.post("/predict", files={"file": fake_file})

    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_without_file():
    """Verifie que /predict renvoie une erreur si aucun fichier n'est envoye"""
    with TestClient(app) as client:
        response = client.post("/predict")
    assert response.status_code == 422