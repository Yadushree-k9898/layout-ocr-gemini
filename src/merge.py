from typing import Dict, Any
from .validate import is_valid

PREF_ORDER = [
    "borrowers", "loan_amount", "recording_date", "recording_location",
    "lender_name", "lender_nmls_id", "broker_name",
    "loan_originator_name", "loan_originator_nmls_id",
]

def merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k in PREF_ORDER:
        b = out.get(k)
        o = overrides.get(k)
        if (not is_valid(k, b)) and is_valid(k, o):
            out[k] = o
    return out
