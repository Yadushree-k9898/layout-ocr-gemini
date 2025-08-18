# Layout-Aware PDF Data Extraction (PaddleOCR + LangChain + Gemini)

Rebuilt on 2025-08-18T07:27:02.775840

This project converts **scanned mortgage/real-estate PDFs** into **structured JSON** using:
- `pdf2image` → `PaddleOCR` (boxes + text)
- Layout-preserving text
- **Gemini** via **LangChain**
- Validation/normalization and merge strategy

## Setup

```bash
python -m venv .venv
# Activate:
#  Windows: .venv\Scripts\activate
#  macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # add your GOOGLE_API_KEY
```

Install **Poppler** (pdf2image requirement):
- Windows: `choco install poppler`
- macOS: `brew install poppler`
- Debian/Ubuntu: `sudo apt-get install -y poppler-utils`

## Run (single PDF)

```bash
python run.py extract tests/samples/your.pdf --out out.json --dpi 300 --model gemini-1.5-flash
```

## Output JSON Keys
`borrowers, loan_amount, recording_date, recording_location, lender_name, lender_nmls_id, broker_name, loan_originator_name, loan_originator_nmls_id`

## Notes
- Increase DPI to 400 for faint stamps or small text.
- Try `gemini-1.5-pro` for tougher docs.
- See `src/validate.py` for regex/date checks and normalization.
