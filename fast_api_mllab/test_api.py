from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"feature1":5,"feature2":10})
    assert response.status_code == 200
    assert "prediction" in response.json()
