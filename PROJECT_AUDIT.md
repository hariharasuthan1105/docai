# PROJECT AUDIT: Document AI System

**Audit Date**: 2026-08-23  
**Auditor**: ML & Production Software Architect  
**Codebase**: `docai`

---

## 1. Executive Summary

This audit assesses the current state of the Document AI repository. The project currently implements a functional, deterministic pipeline utilizing PaddleOCR, regex patterns, fuzzy matching, and basic validation rules. While functional for simple baseline tractor invoices, the system lacks document preprocessing (deskew, denoising), 2D spatial layout intelligence (reading order, label-to-value geometric matching), genuine computer vision signature/stamp detection (currently stubs returning `not_implemented`), calibrated confidence estimation, an API service (FastAPI), comprehensive evaluation benchmarks (ECE, F1, latency), and a visual demo interface.

---

## 2. Current Architecture & Component Audit

```
                              Input Document (PDF/Image/TXT)
                                            │
                                            ▼
                                  PaddleOCR Engine
                            (PaddleOCR 3.7.0 / PaddleX)
                                            │
                                            ▼
                                 OCR Lines & Bounding Boxes
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
                      ▼                                           ▼
             Regex Extraction Engine                     Fuzzy/Exact Matcher
           (HP, Cost, 2-line adjacent)               (RapidFuzz Dealer & Model)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                                   Visual Mark Stubs
                              (Signature / Stamp: stubs)
                                            │
                                            ▼
                                 Business Rules & Scoring
                           (Range checks & weighted confidence)
                                            │
                                            ▼
                                  Structured JSON Output
                                   (FinalDocument Schema)
```

### Detailed Component Review

| Component | File Path | Current Status | Findings & Limitations |
|---|---|---|---|
| **Data Models** | `models/extraction_schema.py` | Working | Defines `FieldValue`, `VisualMark`, `FinalDocument`. Only covers 4 standard fields (`dealer_name`, `model_name`, `horse_power`, `asset_cost`). Lacks support for invoice metadata (`invoice_number`, `invoice_date`, `customer_name`, `customer_address`, `phone_number`, `registration_number`, `serial_number`), extraction method telemetry, and page indices. |
| **OCR Pipeline** | `ocr/paddleocr_engine.py` | Working (Basic) | Wraps PaddleOCR 3.7.0. Handles PDF (via `pypdfium2`) and image inputs. Lacks image preprocessing (deskew, rotation detection, adaptive thresholding, contrast enhancement, denoising) and multi-page layout association. |
| **Field Extraction** | `extraction/regex_engine.py` | Working (Rule-based) | Uses regex passes (single line, 2-line window, numeric fallback) for HP and Cost. Lacks spatial/geometric 2D layout reasoning (identifying horizontal/vertical label-to-value relations, bounding box proximity). |
| **Entity Matching** | `extraction/fuzzy_matcher.py` | Working (Basic) | Uses RapidFuzz for dealer token sort ratio and exact prefix matching for models. Does not return candidate alternatives, nor calculate normalized edit distance / typo tolerance across varied formats. |
| **Validation Engine** | `validation/business_rules.py` | Working (Static) | Static functions for HP and Cost range checks. Not extensible via a registry or modular rule classes. |
| **Confidence Scoring** | `validation/confidence.py` | Heuristic only | Simple weighted average of heuristic extraction scores. Not calibrated against ground truth probabilities (no Platt scaling, isotonic regression, ECE, or Brier score). |
| **Visual Mark Detection** | `vision/signature_detector.py`, `vision/stamp_detector.py` | Stubbed (`not_implemented`) | Returns fixed `VisualMark(status="not_implemented")`. No actual computer vision / contour / HSV / connected-component detection is implemented. |
| **Configuration** | `config.py` | Working | Hardcoded dealer/model lists and thresholds. Needs structured validation profiles and flexible overrides. |
| **CLI** | `cli.py`, `__main__.py` | Working | Basic CLI for single document processing (`python -m docai <file> -o <out>`). Lacks subcommands (`predict`, `batch`, `evaluate`, `benchmark`). |
| **REST API** | *Missing* | Not Implemented | No FastAPI / REST endpoints exist (`/predict`, `/batch`, `/health`, `/schema`). |
| **Evaluation Framework** | *Missing* | Not Implemented | No standard evaluation dataset (`data/samples/`, `data/ground_truth/`), ablation scripts, or robustness benchmarks. |
| **Demo UI** | *Missing* | Not Implemented | No interactive visualization interface (e.g. Streamlit) for document review with bounding boxes and confidence flags. |

---

## 3. Execution & Verification Audit

### Test Suite Execution
- **Command**: `pytest`
- **Result**: 44 tests passed in 9.01s.
- **Coverage**: Covers regex parsing, rapidfuzz matching, business rules, CLI stdout, and pipeline integration with stubbed OCR.

### Sample Document Processing
- **Command**: `python cli.py invoice.png --output output/invoice_result.json`
- **Execution Time**: ~2.5s (Windows CPU)
- **Result**:
  - `dealer_name`: "Mahindra & Mahindra Ltd." (Confidence: 1.00)
  - `model_name`: "Model XYZ" (Confidence: 0.97)
  - `horse_power`: 50.0 HP (Confidence: 0.95)
  - `asset_cost`: 621000.0 (Confidence: 0.92)
  - `dealer_signature`: "not_implemented"
  - `dealer_stamp`: "not_implemented"
  - Overall Confidence: 0.96 (AUTO-APPROVED)

---

## 4. Key Gaps & Broken Elements Identified

1. **No Spatial / Geometric Layout Engine**:
   - The extraction relies entirely on line-by-line regex strings rather than 2D geometry (label on left, value on right; label on top, value below; table grid cell alignment).
2. **Signature & Stamp Detection are Stubs**:
   - `vision/signature_detector.py` and `vision/stamp_detector.py` return `status="not_implemented"`. Real computer vision detectors (HSV color filtering for blue/red stamps, high-density edge/contour variance for pen signatures) must be implemented.
3. **Restricted Field Set**:
   - Only extracts 4 fields. Real-world documents require invoice numbers, invoice dates, customer details, tax/registration numbers, and line items.
4. **Missing Image Preprocessing & Multi-page Geometry**:
   - Documents with rotation (90°/180°/270°), skew, noise, or low contrast are passed directly to OCR without deskewing or adaptive enhancement.
5. **No Probability Calibration**:
   - Confidence scores are arbitrary heuristic floats (0.95, 0.92) without probability calibration (Platt scaling, Isotonic regression, Expected Calibration Error).
6. **No API Server or Interactive UI**:
   - Missing FastAPI service and Streamlit review dashboard.
7. **No Automated Benchmark or Ablation Framework**:
   - Missing ground-truth dataset with evaluation metrics (precision, recall, F1, exact match, normalized edit distance, latency).

---

## 5. Recommended Upgrade Path

1. **Preprocessing & OCR Enhancer**:
   - Add `docai/preprocessing/` with OpenCV-based deskewing, orientation detection, adaptive contrast enhancement, and multi-page coordinate normalization.
2. **Layout & Spatial Reasoning Engine**:
   - Implement `docai/layout/` module with 2D geometric reading order, bounding-box KD-tree / spatial index, label-value association (right-neighbor and below-neighbor queries), and table cell parsing.
3. **Extended Schema & Layered Extractor**:
   - Expand `models/extraction_schema.py` to support 11+ production fields with full method provenance (`layout`, `regex`, `entity_match`, `rule_default`).
4. **Computer Vision Signature & Stamp Detectors**:
   - Implement genuine CV detectors in `docai/vision/` using HSV color space segmentation (blue/red stamp ink) and morphological contour analysis (handwritten signature stroke density).
5. **Extensible Validation & Calibration Engine**:
   - Create a pluggable rule registry in `docai/validation/` and probability calibration via `sklearn.calibration.CalibratedClassifierCV` / Platt scaling with ECE & Brier score reporting.
6. **FastAPI Service & Streamlit UI**:
   - Implement `docai/api/` with `/predict`, `/batch`, `/health`, `/schema` and `docai/demo/` for visual review with interactive bounding boxes.
7. **Evaluation, Ablation & Robustness Suite**:
   - Create synthetic + real test ground-truth dataset, ablation study runner, and document degradation robustness benchmark.
