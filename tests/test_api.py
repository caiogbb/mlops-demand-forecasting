from fastapi.testclient import (
    TestClient,
)

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {"status": "healthy"}


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    assert "model_name" in (response.json())
