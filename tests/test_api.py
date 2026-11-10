from fastapi.testclient import TestClient

from recommendation_engine.api.main import app


def test_health_check(api_model_path):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_recommendations_success(api_model_path):
    with TestClient(app) as client:
        response = client.get("/recommend/44?n=5")

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 44
    assert len(data["recommendations"]) == 5


def test_recommendations_unknown_user(api_model_path):
    with TestClient(app) as client:
        response = client.get("/recommend/9999999999999999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Unknown user id: 9999999999999999"
    }


def test_recommendations_invalid_user_id(api_model_path):
    with TestClient(app) as client:
        response = client.get("/recommend/abc")

    assert response.status_code == 422


def test_recommendations_invalid_n(api_model_path):
    with TestClient(app) as client:
        response = client.get("/recommend/44?n=0")

    assert response.status_code == 422


def test_recommendations_n_exceeds_limit(api_model_path):
    with TestClient(app) as client:
        response = client.get("/recommend/44?n=101")

    assert response.status_code == 422