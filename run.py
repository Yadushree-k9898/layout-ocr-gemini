# cli.py
import typer
from pathlib import Path
import logging
from src.pipeline import run_pipeline, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = typer.Typer(add_completion=False)

@app.command()
def extract(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Input scanned PDF"),
    out: Path = typer.Option(None, help="Where to save the extracted JSON"),
    dpi: int = typer.Option(300, help="DPI for PDF rasterization"),
    model: str = typer.Option("gemini-1.5-flash", help="Gemini model ID")
):
    logging.info(f"📄 Processing: {pdf.name} (dpi={dpi}, model={model})")
    data = run_pipeline(str(pdf), dpi=dpi, model=model)

    if not data:
        logging.error("❌ No data extracted. Check PDF and API settings.")
        raise typer.Exit(code=1)

    if out is None:
        out = pdf.with_suffix(".json")

    save_json(data, str(out))
    logging.info(f"✅ Saved extracted JSON -> {out}")

if __name__ == "__main__":
    app()
