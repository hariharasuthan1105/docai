# Document AI — Production Document Intelligence & Field Extraction

[![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR%203.7-orange.svg)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20REST-teal.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

A high-performance, deterministic Document AI system engineered for invoices, tax documents, and equipment financing applications. Operates with **zero LLM dependencies, zero cloud API costs, and 100% offline reproducibility**.

---

## 🌟 Key Features

* **Multi-Page PDF & Image Preprocessing**: Automatic skew angle correction ($\pm 45^\circ$), projection orientation normalization, adaptive CLAHE contrast enhancement, and bilateral denoising.
* **PaddleOCR PP-OCRv6 Text Extraction**: High-precision text detection and recognition with line-level bounding box tracking.
* **2D Spatial Layout Intelligence**: Geometric neighbor querying, dynamic line band sorting, reading order reconstruction, and key-value anchor pairing.
* **11+ Structured Invoice Fields**: Dealer Name, Model Name, Horse Power, Asset Cost, Invoice Number, Invoice Date, Customer Name, Customer Address, Phone Number, Registration Number, and Serial Number.
* **Computer Vision Visual Mark Verification**: Morphological cursive stroke detection for handwritten signatures and HSV color-space segmentation for official rubber stamps and seals.
* **Extensible Validation Rule Engine**: Pluggable rules for numeric ranges, calendar date formats, telephone standards, and business logic.
* **Statistical Confidence Calibration**: Platt scaling and Isotonic regression calibration with Expected Calibration Error (ECE) minimization.
* **Human-in-the-Loop Decisioning**: Automated audit trail generating `AUTO_APPROVE`, `REVIEW`, or `MANUAL_REVIEW` decisions with explicit reasoning.
* **FastAPI REST Service & Streamlit UI**: Production REST endpoints (`/predict`, `/batch`, `/health`, `/schema`) and interactive visual review dashboard.
* **Benchmarking & Ablation Suite**: Built-in ground-truth dataset generator, field-level F1/NLED metrics, 5-layer ablation experiments, and 6-condition image degradation robustness benchmarks.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/hariharasuthan1105/docai.git
cd docai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # Linux / macOS

# Install production dependencies
pip install -r requirements.txt
```

### 2. Run CLI Extraction

```bash
# Process single invoice document
python -m docai predict invoice.png --output output/invoice_result.json

# Process entire directory
python -m docai batch data/samples/ --output-dir output/batch_results/
```

### 3. Launch FastAPI REST Service

```bash
python -m docai serve --host 0.0.0.0 --port 8000
```
* **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Redoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/samples/sample_01.png", "document_id": "inv_001"}'
```

### 4. Launch Interactive Streamlit Review UI

```bash
python -m docai demo --port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interactively upload invoices, view visual bounding box overlays, and inspect review triggers.

### 5. Run Evaluation & Benchmarking

```bash
# Run ground-truth evaluation metrics (Precision, Recall, F1, NLED, ECE)
python -m docai evaluate data --output EVALUATION_REPORT.md

# Run component ablation study & image degradation benchmarks
python -m docai benchmark data --output BENCHMARK_REPORT.md
```

---

## 📊 System Architecture

```
PDF / Image / Text
        ↓
[ CV Preprocessing: Deskew + CLAHE + Denoise ]
        ↓
[ PaddleOCR PP-OCRv6 Text & Bounding Boxes ]
        ↓
[ 2D Spatial Layout & Geometric Key-Value Parsing ]
        ↓
[ Layered Field Extraction (Layout -> Pattern -> RapidFuzz Matching) ]
        ↓
[ Computer Vision Signature & Rubber Stamp Detection ]
        ↓
[ Extensible Business Rule Validation ]
        ↓
[ Calibrated Confidence Scoring & Human Review Decision ]
        ↓
Structured Document JSON Output
```

---

## 🧪 Test Suite

Run the complete 70-test automated test suite:

```bash
pytest -v
```

---

## 📑 Output Schema

```json
{
  "document_id": "sample_01.png",
  "document": "data/samples/sample_01.png",
  "fields": {
    "dealer_name": {
      "value": "Mahindra & Mahindra Ltd.",
      "confidence": 0.96,
      "bbox": [58.0, 93.0, 269.0, 106.0],
      "source": "fuzzy_match",
      "page": 1
    },
    "model_name": {
      "value": "Arjun Novo 605 DI",
      "confidence": 0.94,
      "bbox": [157.0, 400.0, 277.0, 413.0],
      "source": "exact_match",
      "page": 1
    },
    "horse_power": {
      "value": 50.0,
      "confidence": 0.95,
      "bbox": [597.0, 399.0, 637.0, 413.0],
      "source": "regex",
      "page": 1
    },
    "asset_cost": {
      "value": 550000.0,
      "confidence": 0.92,
      "bbox": [777.0, 618.0, 833.0, 632.0],
      "source": "regex",
      "page": 1
    }
  },
  "dealer_signature": {
    "present": true,
    "confidence": 0.70,
    "bbox": [670.0, 1200.0, 840.0, 1240.0],
    "mark_type": "signature",
    "status": "detected"
  },
  "dealer_stamp": {
    "present": true,
    "confidence": 0.80,
    "bbox": [685.0, 1195.0, 815.0, 1325.0],
    "mark_type": "stamp",
    "status": "detected"
  },
  "overall_confidence": 0.9437,
  "decision": "AUTO_APPROVE",
  "needs_human_review": false,
  "review_reasons": [],
  "processing_time_ms": 3250.0
}
```

---

## 📄 Documentation

* [ARCHITECTURE.md](ARCHITECTURE.md) — Comprehensive system architecture, contracts, and data flows.
* [EVALUATION_REPORT.md](EVALUATION_REPORT.md) — Evaluation benchmark metrics, F1 scores, and ECE calibration error.
* [FINAL_REPORT.md](FINAL_REPORT.md) — Summary of upgrades, benchmarks, and deployment guide.
* [PROJECT_AUDIT.md](PROJECT_AUDIT.md) — Complete repository audit and code evolution.
