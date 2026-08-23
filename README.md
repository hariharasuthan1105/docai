# docai — Deterministic Document AI CLI

A lightweight, **100% deterministic local Document AI application** for extracting structured fields from invoices and equipment documents.

Combines **PaddleOCR**, **robust regex/rule extraction**, **fuzzy dealer matching**, **exact model matching**, **numeric normalization**, **OCR bounding box association**, and **business validation** into a pure command-line interface.

- **Zero LLM / Generative AI dependencies**
- **No API keys or cloud accounts needed**
- **No model weights to download (Ollama, Claude, OpenAI, Gemini are completely absent)**
- **Runs entirely from the local terminal / command prompt**
- **Zero web UI, dashboard, or server overhead**

---

## Architecture

```
                  PDF / Image / Text
                          │
                          ▼
                      PaddleOCR
              (OCR text + bounding boxes)
                          │
                          ▼
                Rule & Regex Extraction
       (HP, Asset Cost, Candidates + OCR line bboxes)
                          │
                          ▼
                Fuzzy Dealer Matching
              (against dealer catalog)
                          │
                          ▼
                 Exact Model Matching
        (against model catalog with normalized text)
                          │
                          ▼
                 Numeric Normalization
         (commas, decimals, Indian numbering)
                          │
                          ▼
                 Business Validation
            (plausible ranges & required fields)
                          │
                          ▼
                 Confidence Scoring
                          │
                          ▼
                  Structured JSON
                          │
                          ▼
                   TERMINAL OUTPUT
```

---

## Target Fields

| Field | Extraction Method | Notes |
|---|---|---|
| **Dealer Name** | Fuzzy Matching | Matched against master dealer catalog with similarity scoring & OCR line bbox |
| **Model Name** | Exact Matching | Matched against master model catalog with OCR text normalization & OCR line bbox |
| **Horse Power** | Regex Extraction | Supports `50 HP`, `50HP`, `50 H.P.`, `Horse Power: 50`, `HP - 50`, `Power: 50` |
| **Asset Cost** | Regex + Normalization | Supports currency symbols (`Rs.`, `₹`, `INR`), Indian commas (`5,50,000`), decimals |
| **Dealer Signature** | Visual Mark Stub | Transparently reports `NOT IMPLEMENTED` (`status: "not_implemented"`) |
| **Dealer Stamp** | Visual Mark Stub | Transparently reports `NOT IMPLEMENTED` (`status: "not_implemented"`) |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <repository_url>
cd docai
```

### 2. Create and activate a virtual environment
```bash
# Create virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## CLI Usage

The primary command is:

```bash
python -m docai <input_file>
```

### Examples:

```bash
# Process a PDF invoice
python -m docai invoice.pdf

# Process an image invoice (PNG / JPG)
python -m docai invoice.png

# Process a pre-extracted text invoice
python -m docai input/sample_invoice.txt

# Export full structured JSON result
python -m docai invoice.pdf --output result.json

# Use custom catalogs JSON
python -m docai invoice.pdf --config custom_catalogs.json

# Specify OCR language (en, hi, gu)
python -m docai invoice.pdf --language hi

# View command-line help
python -m docai --help
```

---

## Command-Line Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `<input_file>` | | Path to document file (PDF, PNG, JPG, or TXT) | *Required* |
| `--output` | `-o` | Save full structured extraction result to a JSON file | None |
| `--config` | `-c` | Optional path to custom JSON file with dealer/model catalogs | Default catalogs |
| `--language` | `-l` | OCR language code: `en` (English), `hi` (Hindi), `gu` (Gujarati) | `en` |
| `--verbose` | `-v` | Show detailed diagnostic logging | `False` |
| `--help` | `-h` | Show help message and exit | |

---

## Example Terminal Output

```text
============================================================
DOCUMENT AI FIELD EXTRACTION
============================================================

Input: input/sample_invoice.txt

[1/5] Running PaddleOCR on input/sample_invoice.txt...
[2/5] Extracting regex fields (Horse Power, Asset Cost, Candidates)...
[3/5] Matching dealer (fuzzy) and model (exact)...
[4/5] Checking signature and stamp marks...
[5/5] Validating business rules & computing confidence...

============================================================
RESULT
============================================================

Dealer Name      : Mahindra Tractors Ltd.
Confidence       : 1.00

Model Name       : Arjun Novo 605 DI
Confidence       : 0.97

Horse Power      : 50.0 HP
Confidence       : 0.95

Asset Cost       : 550000.0
Confidence       : 0.92

Dealer Signature : NOT IMPLEMENTED
Dealer Stamp     : NOT IMPLEMENTED

============================================================
Overall Score    : 0.9600
Decision         : AUTO-APPROVED
============================================================

[✓] Saved structured JSON output to: output/result.json
```

---

## Example JSON Output

When `--output result.json` is specified, the saved JSON conforms to:

```json
{
  "document": "input/sample_invoice.txt",
  "fields": {
    "dealer_name": {
      "value": "Mahindra Tractors Ltd.",
      "confidence": 1.0,
      "bbox": [10.0, 40.0, 350.0, 60.0]
    },
    "model_name": {
      "value": "Arjun Novo 605 DI",
      "confidence": 0.97,
      "bbox": [10.0, 70.0, 300.0, 90.0]
    },
    "horse_power": {
      "value": 50.0,
      "confidence": 0.95,
      "bbox": [10.0, 100.0, 180.0, 120.0]
    },
    "asset_cost": {
      "value": 550000.0,
      "confidence": 0.92,
      "bbox": [10.0, 130.0, 250.0, 150.0]
    },
    "dealer_signature": {
      "status": "not_implemented"
    },
    "dealer_stamp": {
      "status": "not_implemented"
    }
  },
  "validation": {
    "overall_confidence": 0.96,
    "needs_human_review": false,
    "review_reasons": []
  }
}
```

---

## Running Automated Tests

Run the full pytest suite (all tests execute 100% offline in ~1 second):

```bash
pytest -v
```
