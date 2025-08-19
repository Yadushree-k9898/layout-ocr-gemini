# cli.py
import typer
from pathlib import Path
import logging
from src.pipeline import run_pipeline, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer(add_completion=False)


@app.command()
def extract(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Input scanned PDF"),
    out: Path = typer.Option(None, help="Where to save the extracted JSON"),
    dpi: int = typer.Option(300, help="DPI for PDF rasterization"),
    model: str = typer.Option("gemini-1.5-flash", help="Gemini model ID")
):
    """
    Extract structured fields from a scanned mortgage/real-estate PDF.
    """
    try:
        logging.info(f"📄 Processing: {pdf.name} (dpi={dpi}, model={model})")
        data = run_pipeline(str(pdf), dpi=dpi, model=model)

        if not data:
            logging.error("❌ No data extracted. Check PDF and API settings.")
            raise typer.Exit(code=1)

        # Default output: same folder as PDF with .json extension
        if out is None:
            out = pdf.with_suffix(".json")

        save_json(data, str(out))
        logging.info(f"✅ Saved extracted JSON -> {out}")

    except Exception as e:
        logging.error(f"⚠️ Extraction failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
