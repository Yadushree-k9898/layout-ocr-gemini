from typing import List, Dict, Any
from ..preprocess import page_as_layout_text

def pages_to_layout_json(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc = {"pages": []}
    for p in pages:
        doc["pages"].append({
            "page_index": p["page_index"],
            "layout_text": page_as_layout_text(p)
        })
    return doc
