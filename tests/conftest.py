"""Test bootstrap: tiny quick-trained model + throwaway SQLite DB. Env vars MUST be set before importing the app."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="cropai-tests-"))
os.environ.update(DATABASE_URL=f"sqlite:///{_tmp / 'test.db'}", MODEL_DIR=str(_tmp / "models"), RATE_LIMIT_ENABLED="false",
                  SECRET_KEY="test-secret-key-for-pytest-only-0123456789", MODEL_VERSION="latest")

import pytest
from fastapi.testclient import TestClient

from ml.src.quick_train import run as quick_train

quick_train(_tmp / "models", n=1800)

from backend.app.main import app

PAYLOAD = dict(crop="Rice", region="Karnataka", season="Kharif", soil_type="Red", irrigation="Canal", ph=6.2, nitrogen=100, phosphorus=28,
               potassium=120, rainfall_mm=950, temperature_c=27, humidity_pct=72, area_ha=2.0, fertilizer_kg_ha=90, year=2024)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def payload():
    return dict(PAYLOAD)


def _register(client, email, name="Test Farmer"):
    r = client.post("/auth/register", json={"name": name, "email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def farmer(client):
    return _register(client, "farmer1@example.com")


@pytest.fixture(scope="session")
def farmer2(client):
    return _register(client, "farmer2@example.com", "Other Farmer")


@pytest.fixture(scope="session")
def admin(client):
    r = client.post("/auth/login", json={"email": "admin@cropai.app", "password": "Admin@12345"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
