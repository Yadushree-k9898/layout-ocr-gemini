from typing import List, Dict, Any
from dataclasses import dataclass
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import concurrent.futures
import os


os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_dgc"] = "0"
os.environ["FLAGS_use_ngraph"] = "0"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class OCRConfig:
    dpi: int = 300
    lang: str = "en"
    use_angle_cls: bool = True
    use_gpu: bool = False  # Set True if GPU available


class OCRService:
    def __init__(self, cfg: OCRConfig = OCRConfig()):
        """
        Initialize PaddleOCR service with given configuration.
        """
        self.cfg = cfg
        self.ocr = PaddleOCR(
            lang=cfg.lang,
            use_angle_cls=cfg.use_angle_cls,
            show_log=False,
            use_gpu=cfg.use_gpu,
            use_pdserving=False,
            enable_mkldnn=False  # disable oneDNN explicitly
        )
        logging.info("OCRService initialized with config: %s", cfg)

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert a PDF into a list of PIL Images using PyMuPDF.
        """
        images = []
        try:
            doc = fitz.open(pdf_path)
            zoom = self.cfg.dpi / 72  # 72 DPI is default in PDFs
            mat = fitz.Matrix(zoom, zoom)

            for page_num, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

            logging.info("PDF %s converted into %d images", pdf_path, len(images))
        except Exception as e:
            logging.error("Failed to convert PDF %s: %s", pdf_path, str(e))
        return images

    def run_page(self, page_idx: int, img: Image.Image) -> Dict[str, Any]:
        """
        Run OCR on a single page image and return structured results.
        """
        try:
            # Force RGB → Numpy (prevents ONEDNN layout issues)
            np_img = np.array(img.convert("RGB"))

            results = self.ocr.ocr(np_img, cls=True)
            lines = []

            for res in results:
                for line in res:
                    box, (text, score) = line
                    lines.append({
                        "text": text,
                        "score": float(score),
                        "box": box
                    })

            # Sort lines top-to-bottom, then left-to-right
            lines.sort(
                key=lambda l: (min(pt[1] for pt in l["box"]),
                               min(pt[0] for pt in l["box"]))
            )
            return {"page_index": page_idx, "lines": lines}

        except Exception as e:
            logging.warning("OCR failed for page %d: %s", page_idx, str(e))
            return {"page_index": page_idx, "lines": []}

    def run(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Run OCR on all images (pages) concurrently.
        """
        pages: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.run_page, i, img)
                       for i, img in enumerate(images)]

            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="OCR pages"):
                pages.append(future.result())

        # Ensure pages are ordered by index
        pages.sort(key=lambda p: p["page_index"])
        return pages
