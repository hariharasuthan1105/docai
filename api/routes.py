"""
FastAPI Routes for Document AI Service.

Endpoints:
- POST /predict: Process a single document upload or path
- POST /batch: Process multiple documents in batch
- GET /health: Service health and capability diagnostics
- GET /schema: JSON schema of extraction output
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from docai.models.extraction_schema import FinalDocument
from docai.pipeline import DocumentAIPipeline

router = APIRouter()

# Global pipeline instance (lazily initialized or injected)
_pipeline_instance: Optional[DocumentAIPipeline] = None


def get_pipeline() -> DocumentAIPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DocumentAIPipeline()
    return _pipeline_instance


class FilePathRequest(BaseModel):
    file_path: str
    document_id: Optional[str] = "doc_001"


class BatchFilePathRequest(BaseModel):
    file_paths: List[str]


class BatchResponse(BaseModel):
    total_processed: int
    successful: int
    failed: int
    processing_time_ms: float
    results: List[Dict[str, Any]]


@router.get("/health", summary="Healthcheck and diagnostic capabilities")
async def healthcheck() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Document AI API",
        "version": "1.0.0",
        "ocr_engine": "PaddleOCR 3.7",
        "features": {
            "spatial_layout_intelligence": True,
            "computer_vision_signatures": True,
            "computer_vision_stamps": True,
            "confidence_calibration": True,
            "multipage_pdf_support": True,
        },
    }


@router.get("/schema", summary="Extraction JSON schema")
async def get_extraction_schema() -> Dict[str, Any]:
    return FinalDocument.model_json_schema()


@router.post("/predict", summary="Process a single document from file path")
async def predict_document(path_req: FilePathRequest) -> Dict[str, Any]:
    pipeline = get_pipeline()
    if not os.path.exists(path_req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path_req.file_path}")

    doc_id = path_req.document_id or os.path.basename(path_req.file_path)
    try:
        res = pipeline.process(path_req.file_path, document_id=doc_id)
        return res.to_legacy_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}") from e


@router.post("/predict/upload", summary="Process a single document from multipart upload")
async def predict_document_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    pipeline = get_pipeline()
    suffix = os.path.splitext(file.filename or "doc.png")[1] or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        res = pipeline.process(temp_path, document_id=file.filename or "doc")
        return res.to_legacy_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload processing failed: {str(e)}") from e
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/batch", summary="Process multiple documents from file paths")
async def predict_batch(batch_req: BatchFilePathRequest) -> BatchResponse:
    pipeline = get_pipeline()
    start_time = time.perf_counter()
    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0

    for p in batch_req.file_paths:
        try:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Path does not exist: {p}")
            res = pipeline.process(p, document_id=os.path.basename(p))
            results.append(res.to_legacy_dict())
            successful += 1
        except Exception as e:
            failed += 1
            results.append({"document": p, "status": "error", "error": str(e)})

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return BatchResponse(
        total_processed=successful + failed,
        successful=successful,
        failed=failed,
        processing_time_ms=round(elapsed_ms, 2),
        results=results,
    )


@router.post("/batch/upload", summary="Process multiple document uploads")
async def predict_batch_upload(files: List[UploadFile] = File(...)) -> BatchResponse:
    pipeline = get_pipeline()
    start_time = time.perf_counter()
    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0

    for f in files:
        suffix = os.path.splitext(f.filename or "doc.png")[1] or ".png"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(f.file, tmp)
                temp_path = tmp.name
            res = pipeline.process(temp_path, document_id=f.filename or "doc")
            results.append(res.to_legacy_dict())
            successful += 1
        except Exception as e:
            failed += 1
            results.append({"document": f.filename, "status": "error", "error": str(e)})
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return BatchResponse(
        total_processed=successful + failed,
        successful=successful,
        failed=failed,
        processing_time_ms=round(elapsed_ms, 2),
        results=results,
    )
