"""
Integration tests for FastAPI REST Endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from docai.api.app import create_app
from docai.ocr.paddleocr_engine import OCRLine
from docai.pipeline import DocumentAIPipeline
from docai.ocr.paddleocr_engine import StubOCREngine


@pytest.fixture
def client():
    app = create_app()
    # Inject stub pipeline for quick API tests
    from docai.api import routes

    sample_lines = [
        OCRLine(text="Authorised Dealer: Mahindra Tractors Ltd.", bbox=(10, 10, 300, 30), confidence=0.98),
        OCRLine(text="Model: Arjun Novo 605 DI", bbox=(10, 40, 250, 60), confidence=0.98),
        OCRLine(text="Horse Power: 50 HP", bbox=(10, 70, 150, 90), confidence=0.95),
        OCRLine(text="Total Amount: Rs. 5,50,000.00", bbox=(10, 100, 250, 120), confidence=0.92),
    ]
    stub_ocr = StubOCREngine(canned_lines=sample_lines)
    routes._pipeline_instance = DocumentAIPipeline(ocr_engine=stub_ocr)
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "features" in data


def test_schema_endpoint(client):
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "properties" in data
    assert "dealer_name" in data["properties"]


def test_predict_endpoint_file_path(client, tmp_path):
    doc_file = tmp_path / "test_doc.txt"
    doc_file.write_text("Dummy content")

    payload = {"file_path": str(doc_file), "document_id": "test_001"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["document_id"] == "test_001"
    assert "fields" in data
    assert data["fields"]["dealer_name"]["value"] == "Mahindra Tractors Ltd."
    assert data["fields"]["horse_power"]["value"] == 50.0


def test_predict_endpoint_missing_file(client):
    payload = {"file_path": "non_existent_file.pdf"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
