"""
FastAPI Application Factory for Document AI.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docai.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the Document AI FastAPI instance."""
    app = FastAPI(
        title="Document AI API",
        description="Production REST API for Intelligent Document Field Extraction, Spatial Layout Analysis & Computer Vision Verification",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
