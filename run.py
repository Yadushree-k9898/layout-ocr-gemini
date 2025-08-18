import typer
from pathlib import Path
from src.pipeline import run_pipeline, save_json
import logging

logging.basicConfig(level=logging.INFO)
app = typer.Typer(add_completion=False)

@app.command()
def extract(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Input scanned PDF"),
    out: Path = typer.Option("out.json", help="Where to save the extracted JSON"),
    dpi: int = typer.Option(300, help="DPI for PDF rasterization"),
    model: str = typer.Option("gemini-1.5-flash", help="Gemini model ID")
):
    """Extract structured fields from a scanned mortgage/real-estate PDF."""
    data = run_pipeline(str(pdf), dpi=dpi, model=model)
    if data:
        save_json(data, str(out))
        logging.info(f"Saved -> {out}")
    else:
        logging.error("No data extracted. Check PDF and API settings.")

if __name__ == "__main__":
    app()
