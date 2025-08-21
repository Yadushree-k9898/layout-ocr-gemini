from typing import Dict, Any
import logging
import json
from .ocr import OCRService, OCRConfig
from .preprocess import preprocess_pages
from .utils.pdf_utils import pages_to_layout_json
from .gemini_extractor import GeminiExtractor
from .merge import merge
from .validate import normalize

logging.basicConfig(level=logging.INFO)

REQUIRED_FIELDS = [
    "borrowers", "loan_amount", "recording_date", "recording_location",
    "lender_name", "lender_nmls_id", "broker_name",
    "loan_originator_name", "loan_originator_nmls_id",
]

def run_pipeline(pdf_path: str, dpi: int = 300, model: str = "gemini-1.5-flash") -> Dict[str, Any]:
    try:
        logging.info(f"Starting pipeline for: {pdf_path}")

        # 1. OCR
        ocr = OCRService(OCRConfig(dpi=dpi))
        images = ocr.pdf_to_images(pdf_path)
        pages = ocr.run(images)
        logging.info(f"OCR completed. Extracted {len(pages)} pages.")

        # 2. Preprocess
        cleaned = preprocess_pages(pages)
        layout_json = pages_to_layout_json(cleaned)

        # 3. Full extraction
        extractor = GeminiExtractor(model=model)
        full = extractor.extract_full(layout_json)

        # 4. Retry missing fields individually
        missing = [k for k in REQUIRED_FIELDS if full.get(k) in [None, "", [], {}]]
        if missing:
            logging.warning(f"Missing fields detected: {missing}")
            per_field = extractor.extract_fields(layout_json, missing)
            merged = merge(full, per_field)
        else:
            merged = full

        # 5. Normalize
        final = normalize(merged)
        logging.info(f"Pipeline completed successfully for {pdf_path}")
        return final

    except Exception as e:
        logging.error(f"Pipeline failed for {pdf_path}: {e}", exc_info=True)
        return {}

def save_json(data: Dict[str, Any], out_path: str) -> None:
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSON -> {out_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON at {out_path}: {e}")
