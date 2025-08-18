from typing import List, Dict, Any
import re
import logging

logging.basicConfig(level=logging.INFO)

def fix_common_misreads(text: str) -> str:
    t = text
    # Numeric misreads
    t = re.sub(r'(?<=\d)O(?=\d)', '0', t)
    t = re.sub(r'(?<=\$)O(?=\d)', '0', t)
    t = re.sub(r'(?<=NMLS\s*[#:]?\s*)O(?=\d)', '0', t, flags=re.I)
    t = re.sub(r'\bI\b', '1', t)
    t = re.sub(r'\bl\b', '1', t)
    t = re.sub(r'\bS\b', '5', t)
    # Money formatting
    t = re.sub(r'(?<!\$)(\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b)', r'$\1', t)
    # Typography
    t = t.replace('“', '"').replace('”', '"').replace('’', "'").replace('—', '-').replace('–', '-')
    return t.strip()

def preprocess_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_pages = []
    for p in pages:
        new_lines = []
        for l in p["lines"]:
            cleaned_text = fix_common_misreads(l["text"])
            new_lines.append({**l, "text": cleaned_text})
        cleaned_pages.append({**p, "lines": new_lines})
    logging.info(f"Preprocessed {len(cleaned_pages)} pages")
    return cleaned_pages

def page_as_layout_text(page: Dict[str, Any]) -> str:
    chunks = []
    for i, l in enumerate(page["lines"], 1):
        y = min(pt[1] for pt in l["box"])
        x = min(pt[0] for pt in l["box"])
        chunks.append(f"[{i:03d}|y={y:.0f}|x={x:.0f}] {l['text']}")
    return "\n".join(chunks)
