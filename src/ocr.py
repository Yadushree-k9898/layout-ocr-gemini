from typing import List, Dict, Any
from dataclasses import dataclass
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO)

@dataclass
class OCRConfig:
    dpi: int = 300
    lang: str = "en"
    use_angle_cls: bool = True
    use_gpu: bool = False

class OCRService:
    def __init__(self, cfg: OCRConfig = OCRConfig()):
        self.cfg = cfg
        self.ocr = PaddleOCR(
            lang=cfg.lang,
            use_angle_cls=cfg.use_angle_cls,
            show_log=False,
            use_gpu=cfg.use_gpu
        )

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        try:
            images = convert_from_path(pdf_path, dpi=self.cfg.dpi)
            logging.info(f"PDF converted to {len(images)} images")
            return images
        except Exception as e:
            logging.error(f"Failed to convert PDF: {e}")
            return []

    def run_page(self, page_idx: int, img: Image.Image) -> Dict[str, Any]:
        np_img = np.array(img)
        try:
            results = self.ocr.ocr(np_img, cls=True)
            lines = []
            for res in results:
                for line in res:
                    box, (text, score) = line
                    lines.append({"text": text, "score": float(score), "box": box})
            lines.sort(key=lambda l: (min(pt[1] for pt in l["box"]), min(pt[0] for pt in l["box"])))
            return {"page_index": page_idx, "lines": lines}
        except Exception as e:
            logging.warning(f"OCR failed for page {page_idx}: {e}")
            return {"page_index": page_idx, "lines": []}

    def run(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        pages = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.run_page, i, img) for i, img in enumerate(images)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OCR pages"):
                pages.append(future.result())
        pages.sort(key=lambda p: p["page_index"])
        return pages
