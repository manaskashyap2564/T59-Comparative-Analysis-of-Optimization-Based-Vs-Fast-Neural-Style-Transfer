import pytest, sys, os
from unittest.mock import MagicMock

# ── Saare heavy imports ek baar mein mock karo ──────
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.get_device_name.return_value = "cpu"

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['flask_limiter'] = MagicMock()
sys.modules['flask_limiter.util'] = MagicMock()
sys.modules['optimizer_nst'] = MagicMock()
sys.modules['inference'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'

def test_recommend_endpoint(client):
    response = client.get('/api/recommend')
    assert response.status_code == 200
    data = response.get_json()
    assert 'recommended_method' in data

def test_stylize_requires_auth(client):
    response = client.post('/api/stylize')
    assert response.status_code == 401

def test_benchmark_requires_auth(client):
    response = client.post('/api/benchmark')
    assert response.status_code == 401