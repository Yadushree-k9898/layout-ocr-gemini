from typing import Dict, Any

FIELDS = [
    "borrowers",
    "loan_amount",
    "recording_date",
    "recording_location",
    "lender_name",
    "lender_nmls_id",
    "broker_name",
    "loan_originator_name",
    "loan_originator_nmls_id",
]

FIELD_GUIDELINES = {
    "borrowers": "List all borrower names as they appear on the document. Preserve full names.",
    "loan_amount": "Numeric loan amount in dollars, without commas. Example: 475950.00",
    "recording_date": "Recording date in MM/DD/YYYY format.",
    "recording_location": "County and state where recorded. Example: 'Albany County, New York'.",
    "lender_name": "Full legal name of the lender.",
    "lender_nmls_id": "Numeric NMLS ID of the lender (digits only).",
    "broker_name": "Mortgage broker company name, if present.",
    "loan_originator_name": "Full name of the individual loan originator.",
    "loan_originator_nmls_id": "Numeric NMLS ID of the individual loan originator (digits only).",
}

def full_doc_prompt(layout_json: Dict[str, Any]) -> str:
    return f"""
You are an expert at reading US mortgage/real-estate scanned documents.

Extract the following fields from the OCR text and return STRICT JSON only:

{ {field: FIELD_GUIDELINES[field] for field in FIELDS} }

Rules:
- Always return a JSON object with all fields, even if some are null.
- For currency, return numbers only (no '$' or commas).
- For dates, use MM/DD/YYYY.
- For NMLS IDs, return only digits.
- If truly missing, use null.

OCR_LAYOUT_JSON:
{layout_json}
""".strip()

def field_prompt(layout_json: Dict[str, Any], field: str) -> str:
    return f"""
Extract the field "{field}" from the OCR text.

Definition: {FIELD_GUIDELINES.get(field, "No definition provided.")}

Return JSON only in this format:
{{ "{field}": "value or null" }}

OCR_LAYOUT_JSON:
{layout_json}
""".strip()
