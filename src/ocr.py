# src/ocr.py
from typing import List, Dict, Any
from dataclasses import dataclass
import os
import logging
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from tqdm import tqdm
import concurrent.futures
import threading

# ========= Runtime env (must be set BEFORE Paddle initializes) =========
# Disable MKLDNN/oneDNN paths that often cause layout/tensor crashes
os.environ.setdefault("FLAGS_use_mkldnn", "0")
# Keep BLAS backends single-threaded to avoid contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Paddle allocator: safer strategy to reduce fragmentation issues
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
# Also cap Paddle CPU threads
os.environ.setdefault("CPU_NUM", "1")

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class OCRConfig:
    dpi: int = 200                 # keep moderate; very high DPI explodes memory
    lang: str = "en"
    use_angle_cls: bool = True
    workers: int = 1               # PaddleOCR is not thread-safe; keep 1
    max_side: int = 2000           # cap longest side to avoid huge tensors


class OCRService:
    def __init__(self, cfg: OCRConfig = OCRConfig()):
        self.cfg = cfg
        self._ocr_lock = threading.Lock()  # guard all Paddle calls

        # ✅ Instantiate PaddleOCR with only supported arguments
        self.ocr = PaddleOCR(
            lang=cfg.lang,
            use_angle_cls=cfg.use_angle_cls,
        )

        logging.info("OCRService initialized with config: %s", cfg)

    # ---------------------- PDF -> PIL Images ----------------------
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Render each PDF page to a PIL RGB image with DPI scaling."""
        images: List[Image.Image] = []
        try:
            with fitz.open(pdf_path) as doc:
                zoom = self.cfg.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                for _i, page in enumerate(doc, start=1):
                    pix = page.get_pixmap(matrix=mat, alpha=False)  # no alpha
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
            logging.info("PDF %s converted into %d images", pdf_path, len(images))
        except Exception as e:
            logging.error("Failed to convert PDF %s: %s", pdf_path, e)
        return images

    # ---------------------- Helpers ----------------------
    def _prepare_np(self, img: Image.Image) -> np.ndarray:
        """Ensure a safe, contiguous RGB uint8 array with optional downscale."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Downscale very large images
        if self.cfg.max_side and max(img.size) > self.cfg.max_side:
            w, h = img.size
            if w >= h:
                nw = self.cfg.max_side
                nh = max(1, int(round(h * (self.cfg.max_side / w))))
            else:
                nh = self.cfg.max_side
                nw = max(1, int(round(w * (self.cfg.max_side / h))))
            img = img.resize((nw, nh), Image.BILINEAR)

        arr = np.array(img, dtype=np.uint8, copy=False)
        if arr.ndim != 3 or arr.shape[2] != 3:
            arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def _call_paddle(self, arr: np.ndarray, angle_cls: bool) -> Any:
        """Locked Paddle call; arr must be C-contiguous uint8 RGB (H, W, 3)."""
        if arr.size == 0:
            raise ValueError("Empty image array given to OCR.")
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid image tensor for OCR: shape={arr.shape}, dtype={arr.dtype}")
        with self._ocr_lock:
            return self.ocr.ocr(arr, cls=angle_cls)

    def _safe_ocr(self, np_img: np.ndarray):
        """Retry strategy to survive transient Paddle errors."""
        try:
            return self._call_paddle(np_img, self.cfg.use_angle_cls)
        except Exception as e1:
            logging.warning("PaddleOCR attempt#1 failed: %s", e1)

        try:
            return self._call_paddle(np_img.copy(order="C"), self.cfg.use_angle_cls)
        except Exception as e2:
            logging.warning("PaddleOCR attempt#2 (copy) failed: %s", e2)

        try:
            h, w = np_img.shape[:2]
            if h > 1 and w > 1:
                nh = max(1, int(h * 0.8))
                nw = max(1, int(w * 0.8))
                small = Image.fromarray(np_img).resize((nw, nh), Image.BILINEAR)
                small_np = np.ascontiguousarray(np.array(small, dtype=np.uint8))
                return self._call_paddle(small_np, self.cfg.use_angle_cls)
        except Exception as e3:
            logging.warning("PaddleOCR attempt#3 (shrink) failed: %s", e3)

        try:
            return self._call_paddle(np_img, angle_cls=False)
        except Exception as e4:
            logging.error("PaddleOCR attempt#4 (angle_cls=False) failed: %s", e4)

        logging.error("PaddleOCR failed after 4 attempts on page image.")
        return []

    # ---------------------- Per-page OCR ----------------------
    def run_page(self, page_idx: int, img: Image.Image) -> Dict[str, Any]:
        try:
            np_img = self._prepare_np(img)
            results = self._safe_ocr(np_img)

            lines: List[Dict[str, Any]] = []
            if isinstance(results, list) and results:
                for res in results:
                    for line in res:
                        box, txtpack = line[0], line[1]
                        if isinstance(txtpack, (list, tuple)) and len(txtpack) >= 2:
                            text, score = txtpack[0], txtpack[1]
                        else:
                            text, score = str(txtpack), 0.0
                        lines.append({
                            "text": text,
                            "score": float(score),
                            "box": box
                        })

            # Sort lines top-to-bottom, then left-to-right
            lines.sort(
                key=lambda l: (
                    min(pt[1] for pt in l["box"]) if l.get("box") else 0,
                    min(pt[0] for pt in l["box"]) if l.get("box") else 0
                )
            )
            return {"page_index": page_idx, "lines": lines}

        except Exception as e:
            logging.warning("OCR failed for page %d: %s", page_idx, e)
            return {"page_index": page_idx, "lines": []}

    # ---------------------- Batch OCR ----------------------
    def run(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        workers = max(1, int(self.cfg.workers or 1))

        if workers == 1:
            for i, img in tqdm(enumerate(images), total=len(images), desc="OCR pages"):
                pages.append(self.run_page(i, img))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(self.run_page, i, img) for i, img in enumerate(images)]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OCR pages"):
                    pages.append(fut.result())

        pages.sort(key=lambda p: p["page_index"])
        return pages
