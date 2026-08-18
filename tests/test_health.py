"""
Minimal smoke test used by the CI pipeline (.github/workflows/ci.yml).
Run with: pytest tests/
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_import_app():
    """The app module (and model) must load without raising."""
    import main  # noqa: F401
    assert main.app is not None


def test_health_endpoint():
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_api_key():
    from fastapi.testclient import TestClient
    import main

    main.API_KEY = "test-key-123"
    client = TestClient(main.app)
    response = client.post("/predict", json={"prompt": "hello"})
    assert response.status_code == 401
