from pathlib import Path
from src.pipeline import run_pipeline, save_json

# Input and output folders
pdf_folder = Path("Mortgage_PDF")
output_folder = Path("Mortgage_PDF_outputs")
output_folder.mkdir(exist_ok=True)

# Collect all PDF files (case-insensitive)
pdf_files = [f for f in pdf_folder.glob("*") if f.suffix.lower() == ".pdf"]

if not pdf_files:
    print(f"⚠️ No PDF files found in: {pdf_folder.resolve()}")
else:
    for pdf_path in pdf_files:
        try:
            print(f"📄 Processing {pdf_path.name}...")

            # Run pipeline
            data = run_pipeline(str(pdf_path), dpi=300, model="gemini-1.5-flash")

            # Ensure pipeline returns something usable
            if not data:
                print(f"⚠️ No data returned for {pdf_path.name}, skipping...\n")
                continue

            # Save output JSON
            out_path = output_folder / f"{pdf_path.stem}.json"
            save_json(data, str(out_path))

            print(f"✅ Saved JSON -> {out_path}\n")

        except TypeError as te:
            print(f"❌ Function signature mismatch while processing {pdf_path.name}: {te}\n")
        except Exception as e:
            print(f"❌ Unexpected error processing {pdf_path.name}: {e}\n")
