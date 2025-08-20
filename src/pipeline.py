from typing import Dict, Any
import logging
import json
from .ocr import OCRService, OCRConfig
from .preprocess import preprocess_pages
from .utils.pdf_utils import pages_to_layout_json
from .extract import GeminiExtractor
from .merge import merge
from .validate import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fields we want to ensure are extracted
REQUIRED_FIELDS = [
    "borrowers", "loan_amount", "recording_date", "recording_location",
    "lender_name", "lender_nmls_id", "broker_name",
    "loan_originator_name", "loan_originator_nmls_id",
]

def run_pipeline(
    pdf_path: str,
    dpi: int = 300,
    model: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Run the end-to-end pipeline:
      1. Convert PDF to images (using OCRService)
      2. Run OCR on images
      3. Preprocess OCR text
      4. Convert pages into layout-aware JSON
      5. Extract structured data using GeminiExtractor
      6. Handle missing fields with targeted extraction
      7. Normalize output JSON

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution for image conversion (default: 300).
        model (str): Gemini model to use for extraction.

    Returns:
        Dict[str, Any]: Final normalized structured output.
    """
    try:
        logging.info(f"Starting pipeline for: {pdf_path}")

        # Step 1 + 2: PDF -> images -> OCR
        ocr = OCRService(OCRConfig(dpi=dpi))
        images = ocr.pdf_to_images(pdf_path)
        pages = ocr.run(images)
        logging.info(f"OCR completed. Extracted {len(pages)} pages.")

        # Step 3: Preprocess OCR output
        cleaned = preprocess_pages(pages)

        # Step 4: Convert pages to layout JSON
        layout_json = pages_to_layout_json(cleaned)

        # Step 5: Extract structured data using Gemini
        extractor = GeminiExtractor(model=model)
        full = extractor.extract_full(layout_json)

        # Step 6: Handle missing fields if any
        missing = [k for k in REQUIRED_FIELDS if not full.get(k)]
        if missing:
            logging.warning(f"Missing fields detected: {missing}")
            per_field = extractor.extract_fields(layout_json, missing)
            merged = merge(full, per_field)
        else:
            merged = full

        # Step 7: Normalize final result
        final = normalize(merged)
        logging.info(f"Pipeline completed successfully for {pdf_path}")

        return final

    except Exception as e:
        logging.error(f"Pipeline failed for {pdf_path}: {e}", exc_info=True)
        return {}

def save_json(data: Dict[str, Any], out_path: str) -> None:
    """
    Save structured output to a JSON file.

    Args:
        data (Dict[str, Any]): Extracted structured data.
        out_path (str): Path where JSON should be saved.
    """
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSON -> {out_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON at {out_path}: {e}")
