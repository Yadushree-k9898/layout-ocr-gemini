from pathlib import Path
from src.pipeline import run_pipeline, save_json

# Input and output folders
pdf_folder = Path("Mortgage_PDF")
output_folder = Path("Mortgage_PDF_outputs")
output_folder.mkdir(exist_ok=True)

# Collect all PDF files (both .pdf and .PDF)
pdf_files = list(pdf_folder.glob("*.pdf")) + list(pdf_folder.glob("*.PDF"))

if not pdf_files:
    print("⚠️ No PDF files found in Mortgage_PDF folder.")
else:
    for pdf_path in pdf_files:
        try:
            print(f"📄 Processing {pdf_path.name}...")

            # Run pipeline
            data = run_pipeline(str(pdf_path), dpi=300, model="gemini-1.5-flash")

            # Save output JSON
            out_path = output_folder / f"{pdf_path.stem}.json"
            save_json(data, str(out_path))

            print(f"✅ Saved JSON -> {out_path}\n")

        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}\n")
