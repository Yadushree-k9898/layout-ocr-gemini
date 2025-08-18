from typing import Dict, Any

FIELDS = [
    "borrowers", "loan_amount", "recording_date", "recording_location",
    "lender_name", "lender_nmls_id", "broker_name",
    "loan_originator_name", "loan_originator_nmls_id",
]

def full_doc_prompt(layout_json: Dict[str, Any]) -> str:
    return f"""
You are an expert at reading US mortgage/real-estate scanned documents.
Extract fields: {FIELDS} from OCR_LAYOUT_JSON below.
Return STRICT JSON ONLY. Use null if missing. Preserve $ and NMLS IDs as digits.
OCR_LAYOUT_JSON: {layout_json}
""".strip()

def field_prompt(layout_json: Dict[str, Any], field: str) -> str:
    return f"""
Extract ONE field: "{field}" from OCR_LAYOUT_JSON below.
Return JSON only like: {{ "{field}": "value or null" }}
OCR_LAYOUT_JSON: {layout_json}
""".strip()
