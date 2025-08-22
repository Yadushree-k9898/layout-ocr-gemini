Layout OCR with Gemini & LangChain

This project extracts structured information from documents (such as PDFs or images) using OCR (PaddleOCR) and enhances the pipeline with LangChain + Gemini AI for intelligent parsing and field extraction.

It automates:

Converting PDFs/images into text

Detecting document layout and extracting fields

Using LLMs to refine, interpret, and structure the extracted data

🚀 Features

PDF → Image → Text extraction using pdf2image + PaddleOCR

Layout-aware parsing (tables, fields, sections)

AI-powered text refinement with LangChain + Gemini

Configurable prompts for document-level and field-level parsing

CLI-based workflow with progress tracking and logging

📂 Project Structure
layout-ocr-gemini/
│── src/
│   ├── main.py              # Entry point
│   ├── ocr_utils.py         # PDF → Image → Text (OCR pipeline)
│   ├── llm_pipeline.py      # LangChain + Gemini integration
│   ├── prompts/
│   │   ├── full_doc_prompt.py
│   │   └── field_prompt.py
│   ├── config/
│   │   └── settings.py      # API keys, env variables
│── requirements.txt         # Minimal deps
│── requirements-lock.txt    # Full frozen environment
│── README.md

🔧 Installation

Clone the repository:

git clone https://github.com/your-username/layout-ocr-gemini.git
cd layout-ocr-gemini


Create a virtual environment:

python -m venv .venv_ocr
source .venv_ocr/bin/activate   # Linux/Mac
.venv_ocr\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt

⚙️ Configuration

Create a .env file in the project root with:

GOOGLE_API_KEY=your_gemini_api_key

▶️ Usage

Run OCR + AI pipeline on a PDF:

python src/main.py --input docs/sample.pdf --output results.json


Optional CLI arguments:

--input      Path to input PDF
--output     Path to save structured JSON
--fields     Comma-separated list of fields to extract
--verbose    Show debug logs

📊 Example Output

Input: Invoice.pdf
Output (JSON):

{
  "invoice_number": "INV-2025-001",
  "date": "2025-08-21",
  "total_amount": "₹50,000",
  "items": [
    {"name": "Laptop", "qty": 2, "price": "₹25,000"}
  ]
}

🛠️ Tech Stack

OCR: PaddleOCR

PDF Processing: pdf2image, PyMuPDF

AI/LLM: Google Gemini (via langchain-google-genai)

LangChain: Prompt management, structured parsing

Utils: dotenv, pydantic, tqdm, rich

📌 Notes

For Apple Silicon (M1/M2), install paddlepaddle via conda or platform-specific wheel.

Use requirements-lock.txt if you want to reproduce the exact environment.

👩‍💻 Author

Built by Yadushree as part of an OCR + AI pipeline assignment.