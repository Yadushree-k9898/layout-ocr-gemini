from typing import Dict, Any, Optional
import re
import dateparser

MONEY = re.compile(r'\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
NMLS = re.compile(r'\b(\d{1,7})\b')

def _valid_money(v: Optional[str]) -> bool:
    return bool(isinstance(v, str) and MONEY.search(v))

def _valid_nmls(v: Optional[str]) -> bool:
    return bool(isinstance(v, str) and NMLS.fullmatch(v))

def _valid_date(v: Optional[str]) -> bool:
    if not isinstance(v, str):
        return False
    return dateparser.parse(v) is not None

def is_valid(key: str, value: Any) -> bool:
    if key == "loan_amount": return _valid_money(value)
    if key in ("lender_nmls_id", "loan_originator_nmls_id"): return _valid_nmls(value)
    if key == "recording_date": return _valid_date(value)
    if value is None: return False
    if isinstance(value, str) and not value.strip(): return False
    return True

def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    # NMLS digits only
    for k in ["lender_nmls_id", "loan_originator_nmls_id"]:
        v = out.get(k)
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit())
            out[k] = digits if digits else None
    # Standardize loan amount
    if isinstance(out.get("loan_amount"), str):
        m = MONEY.search(out["loan_amount"])
        out["loan_amount"] = m.group(0) if m else out["loan_amount"]
    # Trim strings
    for k, v in out.items():
        if isinstance(v, str): out[k] = v.strip()
    return out
