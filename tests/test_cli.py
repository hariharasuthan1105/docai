"""
Unit tests for Document AI CLI.
"""

import json
from unittest.mock import patch

import pytest
from docai.cli import build_parser, main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Document AI" in captured.out
    assert "predict" in captured.out
    assert "batch" in captured.out


def test_cli_missing_file(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["non_existent_file_xyz.pdf"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Input file not found" in captured.err


def test_cli_run_and_export_json(tmp_path, capsys):
    invoice_path = tmp_path / "invoice.txt"
    invoice_path.write_text(
        "Dealer: Mahindra Tractors Ltd.\n"
        "Model: Arjun Novo 605 DI\n"
        "Engine Power: 50 HP\n"
        "Asset Cost: Rs. 5,50,000.00\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "result.json"

    main([str(invoice_path), "--output", str(output_json)])

    captured = capsys.readouterr()
    assert "EXTRACTION RESULT" in captured.out
    assert "Dealer Name" in captured.out
    assert "Mahindra Tractors Ltd." in captured.out

    # Check generated JSON file
    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["document"] == str(invoice_path)
    assert data["fields"]["dealer_name"]["value"] == "Mahindra Tractors Ltd."
    assert data["fields"]["dealer_name"]["confidence"] >= 0.90
    assert data["fields"]["model_name"]["value"] == "Arjun Novo 605 DI"
    assert data["fields"]["horse_power"]["value"] == 50.0
    assert data["fields"]["asset_cost"]["value"] == 550000.0
    assert data["visual_marks"]["signature"]["status"] == "not_implemented"
    assert data["visual_marks"]["stamp"]["status"] == "not_implemented"
    assert data["review_required"] is False
    assert data["overall_confidence"] >= 0.85


def test_cli_custom_config(tmp_path, capsys):
    cfg_path = tmp_path / "catalogs.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dealers": ["Custom Dealer Corp."],
                "models": ["Custom Model 9000"],
            }
        ),
        encoding="utf-8",
    )
    invoice_path = tmp_path / "invoice.txt"
    invoice_path.write_text(
        "Dealer: Custom Dealer Corp\n"
        "Model: Custom Model 9000\n"
        "Power: 75 HP\n"
        "Cost: Rs. 9,00,000\n",
        encoding="utf-8",
    )

    main([str(invoice_path), "--config", str(cfg_path)])

    captured = capsys.readouterr()
    assert "Custom Dealer Corp" in captured.out
    assert "Custom Model 9000" in captured.out
