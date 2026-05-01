import pytest
import sys
import os

# backend/app.py ko import karne ke liye path set karo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['JWT_SECRET_KEY'] = 'test-secret-key'
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
    assert 'recommendation' in data

# ── Test 3: Stylize bina auth ke reject ho ───────────
def test_stylize_requires_auth(client):
    response = client.post('/api/stylize')
    assert response.status_code == 401

# ── Test 4: Benchmark bina auth ke reject ho ─────────
def test_benchmark_requires_auth(client):
    response = client.get('/api/benchmark')
    assert response.status_code == 401