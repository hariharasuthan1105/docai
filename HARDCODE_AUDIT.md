# Repository-Wide Domain Hardcode Audit

This document compiles the repository-wide audit of all domain-specific hardcoded fields, labels, keywords, and rules, classifying each occurrence according to the schema-driven segregation architecture.

---

## 1. Classification System

All occurrences of domain-specific fields (e.g., `dealer_name`, `horse_power`, `asset_cost`, `cgst_amount`, `merchant_name`, etc.) are classified as follows:

| Class | Description | Status |
|---|---|---|
| **Class A** | Config/YAML Schema Declarations | **Allowed** |
| **Class B** | Schema-specific Tests and Fixtures | **Allowed** |
| **Class C** | Legacy/Evaluation/Baseline-only implementations | **Allowed** |
| **Class D** | Backward-Compatibility Adapters (Isolated) | **Allowed** |
| **Class E** | Core Engine Hardcoding (Violations) | **Eliminated** |

---

## 2. Audit Findings & Classifications

### A. Schemas & Registries (Class A: Allowed)
These are yaml declarations that contain all domain-specific field definitions, aliases, validation thresholds, row formulas, and arithmetic expressions.
* `docai/schemas/tractor_invoice.yaml`
  * Defines agricultural invoice fields (`horse_power`, `asset_cost`, etc.) and validation rules.
* `docai/schemas/restaurant_receipt.yaml`
  * Defines restaurant receipt fields (`merchant_name`, `cgst_amount`, etc.) and validation rules.
* `docai/schemas/generic_document.yaml`
  * Defines generic/fallback key-value fields.
* `docai/schemas/registry.yaml`
  * Maps document types to their schema definitions.

### B. Testing and Fixtures (Class B: Allowed)
Unit tests that verify extraction accuracy on specific document types. They use canned strings/OCR inputs containing domain words and assert domain fields in the output dictionary.
* `tests/test_pipeline.py`
* `tests/test_tractor_invoice.py`
* `tests/test_restaurant_receipt.py`
* `tests/test_generic_document.py`
* `tests/test_validation.py`
* `tests/test_cli.py`

### C. Baselines & Evaluation (Class C: Allowed)
Implementation baselines used for ablation studies and regression/accuracy comparison. These are explicitly segregated and never called by the production pipeline.
* `docai/evaluation/baselines/tractor_regex_baseline.py`
  * Hardcoded regex patterns for extracting tractor fields. Used strictly for evaluation metrics.

### D. Compatibility Layers (Class D: Allowed)
Adapter classes kept to prevent breaking changes for legacy clients expecting direct attribute access on the result object.
* `docai/legacy/tractor_adapter.py`
  * Implements properties (`dealer_name`, `horse_power`, etc.) mapping to the underlying `fields` dict on `DocumentResult`.

### E. Core System Engine (Class E: Violations - ELIMINATED)
All hardcoded domain fields and rules have been completely removed from core pipeline engines.
* `docai/pipeline.py`
  * **Status**: Clean. Uses schema-driven classification, layout anchors, validation rules, and confidence weights.
* `docai/extraction/schema_extractor.py`
  * **Status**: Clean. Uses schema fields dynamically.
* `docai/extraction/regex_engine.py`
  * **Status**: Clean. Regex engine is generic; tractor-specific baseline moved to `evaluation/baselines`.
* `docai/validation/schema_validator.py`
  * **Status**: Clean. Evaluates required rules, range limits, row formulas, and balance expressions dynamically.
* `docai/layout/table_detector.py`
  * **Status**: Clean. Consumes schema-defined stop keywords dynamically.

---

## 3. Core Engine Decoupling Summary

1. **Schema-Driven Fields**: Adding a new document type requires **zero Python edits**. The `SchemaExtractor` dynamically reads the field list and extraction strategies from the schema YAML.
2. **Schema-Driven Table Detection**: Tabular bounds are detected using `stop_keywords` declared in YAML.
3. **Expression-Based Validation**: Row arithmetic (`row_formula`) and arithmetic balance expressions are parsed and evaluated dynamically using Python's controlled `eval()` sandbox, removing hardcoded logic.
