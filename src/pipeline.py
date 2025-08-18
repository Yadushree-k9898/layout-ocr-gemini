from typing import Dict, Any
from .ocr import OCRService, OCRConfig
from .preprocess import preprocess_pages
from .utils.pdf_utils import pages_to_layout_json
from .extract import GeminiExtractor
from .merge import merge
from .validate import normalize
import logging

logging.basicConfig(level=logging.INFO)

REQUIRED_FIELDS = [
    "borrowers", "loan_amount", "recording_date", "recording_location",
    "lender_name", "lender_nmls_id", "broker_name",
    "loan_originator_name", "loan_originator_nmls_id",
]

def run_pipeline(pdf_path: str, dpi: int = 300, model: str = "gemini-1.5-flash") -> Dict[str, Any]:
    try:
        ocr = OCRService(OCRConfig(dpi=dpi))
        images = ocr.pdf_to_images(pdf_path)
        pages = ocr.run(images)

        cleaned = preprocess_pages(pages)
        layout_json = pages_to_layout_json(cleaned)

        extractor = GeminiExtractor(model=model)
        full = extractor.extract_full(layout_json)

        missing = [k for k in REQUIRED_FIELDS if not full.get(k)]
        if missing:
            per_field = extractor.extract_fields(layout_json, missing)
            merged = merge(full, per_field)
        else:
            merged = full

        final = normalize(merged)
        logging.info("Pipeline completed successfully")
        return final
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return {}

def save_json(data: Dict[str, Any], out_path: str):
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
