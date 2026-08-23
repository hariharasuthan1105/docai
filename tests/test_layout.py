"""
Unit tests for 2D Spatial Layout Intelligence & Key-Value Extraction.
"""

from docai.layout.kv_extractor import LayoutKVExtractor
from docai.layout.spatial_index import BoundingBox, SpatialIndex
from docai.ocr.paddleocr_engine import OCRLine


def test_bounding_box_geometry():
    box1 = BoundingBox(10.0, 10.0, 50.0, 30.0)
    assert box1.width == 40.0
    assert box1.height == 20.0
    assert box1.center_x == 30.0
    assert box1.center_y == 20.0

    box2 = BoundingBox(10.0, 10.0, 50.0, 30.0)
    assert box1.iou(box2) == 1.0

    box3 = BoundingBox(100.0, 100.0, 150.0, 150.0)
    assert box1.iou(box3) == 0.0


def test_spatial_index_reading_order():
    # Items in disordered order
    lines = [
        OCRLine(text="Line 2 Left", bbox=(10.0, 50.0, 100.0, 70.0), confidence=0.9),
        OCRLine(text="Line 1 Right", bbox=(150.0, 10.0, 250.0, 30.0), confidence=0.9),
        OCRLine(text="Line 1 Left", bbox=(10.0, 10.0, 100.0, 30.0), confidence=0.9),
        OCRLine(text="Line 2 Right", bbox=(150.0, 50.0, 250.0, 70.0), confidence=0.9),
    ]

    index = SpatialIndex()
    sorted_lines = index.sort_reading_order(lines)

    assert len(sorted_lines) == 4
    assert sorted_lines[0].text == "Line 1 Left"
    assert sorted_lines[1].text == "Line 1 Right"
    assert sorted_lines[2].text == "Line 2 Left"
    assert sorted_lines[3].text == "Line 2 Right"


def test_spatial_index_neighbor_search():
    label_item = OCRLine(text="Horse Power:", bbox=(10.0, 100.0, 120.0, 120.0), confidence=0.95)
    val_item = OCRLine(text="50 HP", bbox=(140.0, 100.0, 200.0, 120.0), confidence=0.95)
    other_item = OCRLine(text="Footer Notes", bbox=(10.0, 500.0, 100.0, 520.0), confidence=0.9)

    index = SpatialIndex([label_item, val_item, other_item])
    right = index.find_right_neighbor(label_item)

    assert right is not None
    matched_item, score = right
    assert matched_item.text == "50 HP"
    assert score > 0.8


def test_layout_kv_extractor():
    lines = [
        OCRLine(text="Horse Power:", bbox=(10.0, 100.0, 120.0, 120.0), confidence=0.95),
        OCRLine(text="50 HP", bbox=(140.0, 100.0, 200.0, 120.0), confidence=0.95),
        OCRLine(text="Total (₹)", bbox=(10.0, 200.0, 100.0, 220.0), confidence=0.95),
        OCRLine(text="732,780.00", bbox=(10.0, 230.0, 120.0, 250.0), confidence=0.95),
    ]

    extractor = LayoutKVExtractor()
    fields = extractor.extract_fields(lines)

    assert "horse_power" in fields
    assert "50 HP" in str(fields["horse_power"].value)

    assert "asset_cost" in fields
    assert "732,780.00" in str(fields["asset_cost"].value)
