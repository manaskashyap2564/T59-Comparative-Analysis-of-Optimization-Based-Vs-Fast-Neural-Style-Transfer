import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Mock karo BEFORE app import
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False          # bool return karo

mock_torch.cuda.get_device_name.return_value = "cpu"       # string return karo

sys.modules['torch'] = mock_torch
sys.modules['torchvision'] = MagicMock()
sys.modules['flask_limiter'] = MagicMock()
sys.modules['flask_limiter.util'] = MagicMock()
sys.modules['optimizer_nst'] = MagicMock()
sys.modules['inference'] = MagicMock()
sys.modules['load_checkpoint'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ── Test 1: Health endpoint ──────────────────────────
def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'

# ── Test 2: Recommend endpoint ───────────────────────
def test_recommend_endpoint(client):
    response = client.get('/api/recommend')
    assert response.status_code == 200
    data = response.get_json()
    assert 'recommended_method' in data

# ── Test 3: Stylize bina auth ke reject ho ───────────
def test_stylize_requires_auth(client):
    response = client.post('/api/stylize')
    assert response.status_code == 401

# ── Test 4: Benchmark bina auth ke reject ho ─────────
def test_benchmark_requires_auth(client):
    response = client.post('/api/benchmark')
    assert response.status_code == 401