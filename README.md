# README: Layout-Aware PDF Data Extraction Pipeline

## Project Overview

This project implements a workflow to extract structured data from scanned mortgage and real estate PDFs using layout-aware OCR and AI-driven extraction. The system converts PDFs to images, applies OCR with bounding boxes, processes the extracted text, and leverages LangChain and Gemini AI to produce accurate, structured JSON outputs.

## Features

* Converts multi-page PDFs to high-resolution images.
* Layout-aware OCR using PaddleOCR.
* Preprocessing to fix common OCR misreads.
* AI-based field extraction with full-document and per-field retries.
* Validation and normalization of extracted data.
* Merge strategy to maximize data accuracy.

## Requirements

* Python 3.10+
* Dependencies (see `requirements.txt`):

  * paddleocr
  * paddlepaddle
  * pdf2image
  * PyMuPDF
  * LangChain
  * google-generativeai
  * python-dotenv
  * pydantic
  * numpy
  * regex
  * dateparser
  * tqdm

## Installation

```bash
git clone <repo_url>
cd pdf-extraction-pipeline
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

2. Place PDF files to be processed in the `Mortgage_PDF` folder.

## Usage

```bash
python main.py
```

* The script processes all PDFs in `Mortgage_PDF`.
* Output JSON files are saved in `Mortgage_PDF_outputs`.

## Folder Structure

```
Mortgage_PDF/              # Input PDFs
Mortgage_PDF_outputs/      # Output JSON files
src/                       # Source code modules
  pipeline.py              # Orchestrates OCR + AI extraction
  ocr.py                   # PDF to image + OCR
  preprocess.py            # Preprocessing and misread fixes
  gemini_extractor.py      # AI extraction using Gemini
  merge.py                 # Merges extraction results
  validate.py              # Validation & normalization functions
```

## Workflow

1. **PDF to Images**: Convert PDF pages to images using `pdf2image` and PyMuPDF.
2. **OCR**: Extract text and layout using PaddleOCR.
3. **Preprocessing**: Clean OCR text and fix common misreads.
4. **AI Extraction**: Use LangChain + Gemini AI for structured extraction.
5. **Validation & Normalization**: Ensure correct formats for dates, amounts, and IDs.
6. **Merge Results**: Combine full-document extraction with field retries.
7. **Output**: Save structured JSON per PDF.

## Example Output

```json
{
 "borrowers": "Elizabeth Howerton and Travis Howerton (spouses)",
 "loan_amount": "$475,950.00",
 "recording_date": "April 1, 2025",
 "recording_location": "Albany County Clerk's Office",
 "lender_name": "US Mortgage Corporation",
 "lender_nmls_id": "3901",
 "broker_name": null,
 "loan_originator_name": "Willam John Lane",
 "loan_originator_nmls_id": "65175"
}
```

## Notes

* Ensure `GEMINI_API_KEY` is valid and has access to the Gemini AI Free Tier.
* Preprocessing handles common OCR errors like `O -> 0` and `I -> 1`.
* Layout-aware OCR significantly improves reading order and extraction accuracy.

## License

This project is open-source and available under the MIT License.
