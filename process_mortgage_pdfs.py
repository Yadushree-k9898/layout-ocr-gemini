from pathlib import Path
from src.pipeline import run_pipeline, save_json


pdf_folder = Path("Mortgage_PDF")
output_folder = Path("Mortgage_PDF_outputs")
output_folder.mkdir(exist_ok=True)


for pdf_path in pdf_folder.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    
   
    data = run_pipeline(str(pdf_path), dpi=300, model="gemini-1.5-flash")
    
   
    out_path = output_folder / f"{pdf_path.stem}.json"
    save_json(data, str(out_path))
    print(f"Saved JSON -> {out_path}")
