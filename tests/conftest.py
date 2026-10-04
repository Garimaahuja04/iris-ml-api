import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_utils import MODEL_PATH
from train_model import train


@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Train the model once if the .joblib file is missing."""
    if not MODEL_PATH.exists():
        train()


@pytest.fixture()
def client():
    return TestClient(app)
