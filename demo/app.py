"""
Interactive Streamlit Dashboard for Document AI.

Run with:
    streamlit run docai/demo/app.py
or
    python -m docai demo
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from docai.models.extraction_schema import FinalDocument, ReviewDecision
from docai.pipeline import DocumentAIPipeline
from docai.preprocessing.pdf_handler import render_pdf_to_images


def draw_bounding_boxes(image: Image.Image, final_doc: FinalDocument) -> Image.Image:
    """Draw bounding boxes with field labels onto the document image."""
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    # Color palette
    field_color = (0, 120, 255)
    sig_color = (34, 197, 94)
    stamp_color = (239, 68, 68)

    # Draw fields
    all_fields = final_doc.get_all_fields()
    for fname, fval in all_fields.items():
        if fval.is_present() and fval.bbox and len(fval.bbox) == 4:
            x0, y0, x1, y1 = fval.bbox
            draw.rectangle([x0, y0, x1, y1], outline=field_color, width=3)
            label = f"{fname}: {fval.value} ({fval.confidence:.2f})"
            draw.text((x0, max(0, y0 - 12)), label, fill=field_color)

    # Draw signature
    if final_doc.dealer_signature.present and final_doc.dealer_signature.bbox:
        x0, y0, x1, y1 = final_doc.dealer_signature.bbox
        draw.rectangle([x0, y0, x1, y1], outline=sig_color, width=4)
        draw.text((x0, max(0, y0 - 14)), f"Signature ({final_doc.dealer_signature.confidence:.2f})", fill=sig_color)

    # Draw stamp
    if final_doc.dealer_stamp.present and final_doc.dealer_stamp.bbox:
        x0, y0, x1, y1 = final_doc.dealer_stamp.bbox
        draw.rectangle([x0, y0, x1, y1], outline=stamp_color, width=4)
        draw.text((x0, max(0, y0 - 14)), f"Stamp ({final_doc.dealer_stamp.confidence:.2f})", fill=stamp_color)

    return img_draw


def main():
    st.set_page_config(
        page_title="Document AI - Production Field Extractor",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Document AI — Field Extraction & Layout Intelligence")
    st.markdown(
        "Production-grade, 100% deterministic Document AI engine with spatial layout analysis, "
        "computer vision mark verification, and human-in-the-loop decisioning."
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    ocr_lang = st.sidebar.selectbox("OCR Language", ["en", "hi", "gu"], index=0)
    enable_preprocess = st.sidebar.checkbox("Enable CV Preprocessing (Deskew/CLAHE)", value=True)
    auto_approve_thresh = st.sidebar.slider("Auto-Approve Threshold", 0.50, 1.00, 0.85, 0.05)

    @st.cache_resource
    def load_pipeline(lang: str, preprocess: bool):
        return DocumentAIPipeline(ocr_lang=lang, enable_preprocessing=preprocess)

    pipeline = load_pipeline(ocr_lang, enable_preprocess)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Invoice / Equipment Document (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🖼️ Document Preview")
            if suffix.lower() == ".pdf":
                pdf_pages = render_pdf_to_images(temp_path, max_pages=1)
                display_img = Image.fromarray(cv2.cvtColor(pdf_pages[0].image, cv2.COLOR_BGR2RGB)) if pdf_pages else None
            else:
                display_img = Image.open(temp_path)

            if display_img:
                st.image(display_img, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.subheader("⚡ Processing")
            if st.button("🚀 Run Extraction Pipeline", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(msg: str, step: int, total: int):
                    progress_bar.progress(step / float(total))
                    status_text.text(f"[{step}/{total}] {msg}")

                start = time.perf_counter()
                final_doc = pipeline.process(
                    temp_path,
                    document_id=uploaded_file.name,
                    on_progress=update_progress,
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                progress_bar.progress(1.0)
                status_text.success(f"Processing completed in {elapsed:.0f} ms!")

                # Decision Banner
                st.markdown("---")
                dec_col1, dec_col2, dec_col3 = st.columns(3)
                with dec_col1:
                    st.metric("Overall Confidence", f"{final_doc.overall_confidence:.2%}")
                with dec_col2:
                    if final_doc.decision == ReviewDecision.AUTO_APPROVE:
                        st.success("✅ DECISION: AUTO-APPROVED")
                    elif final_doc.decision == ReviewDecision.REVIEW:
                        st.warning("⚠️ DECISION: REVIEW REQUIRED")
                    else:
                        st.error("🚨 DECISION: MANUAL REVIEW REQUIRED")
                with dec_col3:
                    st.metric("Processing Latency", f"{elapsed:.0f} ms")

                # Review Reasons
                if final_doc.review_reasons:
                    st.info("📋 **Review Reasons / Audit Trail:**\n" + "\n".join(f"- {r}" for r in final_doc.review_reasons))

                # Extracted Fields
                st.subheader("📊 Extracted Structured Fields")
                table_data = []
                for fname, fval in final_doc.get_all_fields().items():
                    table_data.append({
                        "Field": fname.replace("_", " ").title(),
                        "Value": str(fval.value) if fval.is_present() else "—",
                        "Confidence": f"{fval.confidence:.2%}",
                        "Method": fval.method,
                        "Status": "✅ " + fval.validation_status if fval.validation_status == "valid" else "⚠️ " + fval.validation_status,
                    })
                st.table(table_data)

                # Visual Marks
                st.subheader("🖋️ Visual Mark Verification")
                vm_col1, vm_col2 = st.columns(2)
                with vm_col1:
                    sig_status = "✅ Detected" if final_doc.dealer_signature.present else "❌ Not Found"
                    st.write(f"**Dealer Signature:** {sig_status} ({final_doc.dealer_signature.confidence:.2%})")
                with vm_col2:
                    stamp_status = "✅ Detected" if final_doc.dealer_stamp.present else "❌ Not Found"
                    st.write(f"**Dealer Stamp:** {stamp_status} ({final_doc.dealer_stamp.confidence:.2%})")

                # Visual bounding box overlay
                if display_img:
                    st.subheader("🎯 Visual Bounding Box Overlay")
                    overlay_img = draw_bounding_boxes(display_img, final_doc)
                    st.image(overlay_img, caption="Detected Bounding Boxes", use_container_width=True)

                # Download JSON
                st.download_button(
                    "💾 Download Extraction JSON",
                    data=json.dumps(final_doc.to_legacy_dict(), indent=2),
                    file_name=f"{uploaded_file.name}_result.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
