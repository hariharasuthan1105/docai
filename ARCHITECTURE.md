# Document AI System Architecture

## 1. Overview & Architectural Philosophy

The **Document AI Engine** is a high-performance, deterministic document intelligence system engineered specifically for financial invoices, tax documents, and heavy machinery equipment financing applications.

Unlike generative LLM-based pipelines that introduce nondeterminism, hallucination risk, token cost, and network latency, this system relies on:
1. **Intelligent Computer Vision Preprocessing**: Text contour deskewing, projection orientation normalization, adaptive CLAHE contrast enhancement, and bilateral edge-preserving denoising.
2. **State-of-the-Art OCR Extraction**: PaddleOCR PP-OCRv6 deep learning detection and recognition with multi-page spatial indexing.
3. **2D Spatial Layout Intelligence**: Geometric neighbor querying, reading order sorting, and key-value anchor alignment.
4. **Layered Field Extraction**: Multi-pass consensus combining spatial layout parsing, regex pattern extraction, and catalog entity matching.
5. **Computer Vision Visual Mark Verification**: Morphological cursive stroke detection for handwritten signatures and HSV color-space segmentation for official rubber stamps and seals.
6. **Extensible Business Rule Engine**: Domain validation and consistency checks.
7. **Multi-Signal Calibrated Confidence & Human-in-the-Loop Decisioning**: Platt scaling and Isotonic regression calibration with transparent review triggers.

---

## 2. End-to-End Pipeline Architecture

```
                       [ Input Document: PDF / Image / Text ]
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │         Component 1: Computer Vision Preprocessing     │
            │  • Multi-page PDF Rendering (pypdfium2)                │
            │  • Text Contour Skew Angle Detection & Correction      │
            │  • Horizontal/Vertical Projection Orientation Check    │
            │  • CLAHE Dynamic Contrast & Bilateral Denoising        │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │          Component 2: PaddleOCR PP-OCRv6 Engine        │
            │  • Text Detection (DBNet / PP-OCRv6 Det)               │
            │  • Textline Orientation Classification                 │
            │  • Character Recognition (SVTR / PP-OCRv6 Rec)         │
            │  • Per-line Bounding Box & Confidence Computation      │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │         Component 3: 2D Spatial Layout Intelligence    │
            │  • Reading-Order Sorting via Vertical Overlap Bands    │
            │  • Horizontal Right-Neighbor Geometric Queries         │
            │  • Vertical Below-Neighbor Geometric Queries           │
            │  • Semantic Key-Value Anchor Pairing (KVExtractor)     │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │         Component 4: Layered Multi-Strategy Extraction │
            │  • Level 1: Spatial Layout Key-Values                  │
            │  • Level 2: Deterministic Regex Patterns               │
            │  • Level 3: RapidFuzz Catalog Entity Matching          │
            │  • Level 4: Consensus & Field Provenance Reconciliation│
            └────────────────────────────┬───────────────────────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     │                                       │
                     ▼                                       ▼
    ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
    │  Component 5: Vision Mark Detect │   │ Component 6: Business Validation │
    │ • Handwritten Pen Stroke Contour │   │ • Extensible Rule Engine         │
    │ • HSV Rubber Stamp Segmentation  │   │ • Numeric Range Integrity (HP/₹) │
    │ • Spatial Seal Geometry Filtering│   │ • Calendar Date Format & Range   │
    └──────────────────┬───────────────┘   │ • Mobile Phone Number Format     │
                       │                   └──────────────────┬───────────────┘
                       │                                      │
                       └───────────────────┬──────────────────┘
                                           │
                                           ▼
            ┌────────────────────────────────────────────────────────┐
            │      Component 7: Confidence Calibration & Decision    │
            │  • Multi-Signal Score Blending                         │
            │  • Statistical Probability Calibration (Platt/Isotonic)│
            │  • Expected Calibration Error (ECE) Minimization       │
            │  • Decision: AUTO_APPROVE / REVIEW / MANUAL_REVIEW     │
            │  • Explainable Human Review Triggers & Audit Trail     │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                 [ Output: Standardized Structured Document Schema ]
```

---

## 3. Core Component Contracts

### 3.1 Data Models (`docai/models/extraction_schema.py`)
- **`FieldValue`**: Holds `value`, `confidence` $\in [0, 1]$, `bbox: [x0, y0, x1, y1]`, `source: FieldSource`, `source_text`, `evidence`, `page: int`, `validation_status: str`, and `method: str`.
- **`VisualMark`**: Tracks `present: bool`, `bbox: [x0, y0, x1, y1]`, `confidence: float`, `mark_type: str`, `status: str`, and `details: dict`.
- **`FinalDocument`**: Full document representation with 11 extracted fields, signature and stamp marks, overall confidence, calibrated confidence, human review decision (`AUTO_APPROVE`, `REVIEW`, `MANUAL_REVIEW`), review reasons, and execution latency.

### 3.2 2D Spatial Indexing (`docai/layout/spatial_index.py`)
- Calculates Intersection over Union (IoU), horizontal distance, and vertical overlap.
- Groups text lines into dynamic reading bands and sorts top-to-bottom, left-to-right.
- Employs spatial ray casting to identify label-to-value associations across tabular columns and adjacent key-value pairs.

### 3.3 Genuine Computer Vision Detectors (`docai/vision/`)
- **`CVSignatureDetector`**: Performs morphological gradient filtering to isolate high-curvature cursive handwritten strokes from uniform printed font blocks and straight table gridlines.
- **`CVStampDetector`**: Segments characteristic blue, red, and magenta rubber stamp inks in HSV space, computing circularity and pixel density to detect official seals.

### 3.4 Confidence Calibration (`docai/validation/confidence.py`)
- Implements Platt scaling (logistic regression) and Isotonic regression to transform heuristic scoring weights into empirical error probabilities.
- Computes Expected Calibration Error (ECE) and Brier scores to guarantee that a score of 0.90 corresponds to a 90% empirical accuracy rate.

---

## 4. Performance & Scalability

| Metric | Specification / Measured Benchmark |
|---|---|
| **Latency per Page** | 1.8s - 3.5s (CPU) / < 300ms (GPU) |
| **Throughput** | Parallel batch processing via FastAPI async workers |
| **Model Footprint** | ~15MB (PP-OCRv6 lightweight weights) |
| **External Dependencies** | 0 cloud APIs, 0 token costs, 100% air-gapped |
