# extractor/extract.py
from typing import Dict, Any, List
from .gemini_extractor import GeminiExtractor

class ExtractorWrapper:
    """
    Optional wrapper to allow switching between multiple extractors.
    """
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.extractor = GeminiExtractor(model=model)

    def extract_document(self, layout_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.extractor.extract_full(layout_json)

    def extract_missing_fields(self, layout_json: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        return self.extractor.extract_fields(layout_json, fields)
