"""
RapidOCR Engine Module
Manages Detection and Recognition engines for OCR processing.
Based on the optimal configuration from rapidocr_test_gui.py.
"""

from typing import Any, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    from rapidocr_onnxruntime import RapidOCR
except ModuleNotFoundError:
    from rapidocr import RapidOCR


def get_rotate_crop_image(img: np.ndarray, points) -> np.ndarray:
    """
    Manually crops and warps the image based on detection points.
    Uses INTER_CUBIC and BORDER_REPLICATE to maintain quality for recognition.

    Args:
        img: Input image as numpy array
        points: Detection points (4 corner points)

    Returns:
        Warped/cropped image as numpy array
    """
    points = np.array(points, dtype=np.float32)

    # Sort points to strictly ensure: [top-left, top-right, bottom-right, bottom-left]
    x_sorted = points[np.argsort(points[:, 0]), :]
    left_most, right_most = x_sorted[:2, :], x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    points = np.array([tl, tr, br, bl], dtype=np.float32)

    # Determine target image dimensions
    width_A = np.linalg.norm(br - bl)
    width_B = np.linalg.norm(tr - tl)
    max_width = max(int(width_A), int(width_B))

    height_A = np.linalg.norm(tr - br)
    height_B = np.linalg.norm(tl - bl)
    max_height = max(int(height_A), int(height_B))

    dst_pts = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(points, dst_pts)

    # Crucial: Use INTER_CUBIC for better resizing quality
    # and BORDER_REPLICATE to avoid black edges interfering with text
    warped = cv2.warpPerspective(
        img,
        M,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


class RapidOCREngine:
    """
    Wrapper for RapidOCR with separate Detection and Recognition engines.
    Implements the manual Det -> Crop -> Rec pipeline for optimal results.
    """

    def __init__(self, language: str = "Korean", retry_low_confidence: bool = False):
        self.det_engine: Optional[RapidOCR] = None
        self.rec_engine: Optional[RapidOCR] = None
        self._fallback_engine: Optional[RapidOCR] = None
        self.language = language
        self.retry_low_confidence = retry_low_confidence
        self._initialize_engines()

    def _get_rec_model_and_dict(self, language: str) -> tuple[str, str]:
        """Get recognition model and dictionary paths based on language."""
        # Use absolute paths relative to this file (app/core/rapid_ocr_engine.py)
        # Go up 2 levels: app/core -> app -> EasyScanlate
        base_dir = Path(__file__).parent.parent.parent
        
        if language == "Korean":
            return (
                str(base_dir / "OCR/model/korean_PP-OCRv5_rec_mobile_infer.onnx"),
                str(base_dir / "OCR/dict/korean_dict.txt")
            )
        else:
            return (
                str(base_dir / "OCR/model/ch_PP-OCRv5_rec_mobile_infer.onnx"),
                str(base_dir / "OCR/dict/ppocrv5_dict.txt")
            )

    def set_language(self, language: str):
        """Re-initialize recognition engine with a different language."""
        self.language = language
        self._initialize_engines()
        self._fallback_engine = None  # reset fallback when primary changes

    def _init_fallback_engine(self):
        """Lazy-init the secondary language recognition engine for dual-pass."""
        if hasattr(self, "_fallback_engine") and self._fallback_engine is not None:
            return
        fallback_lang = "Chinese" if self.language == "Korean" else "Korean"
        base_dir = Path(__file__).parent.parent.parent
        rec_model_path, rec_keys_path = self._get_rec_model_and_dict(fallback_lang)
        if Path(rec_model_path).exists():
            self._fallback_engine = RapidOCR(
                rec_model_path=rec_model_path,
                rec_keys_path=rec_keys_path,
                use_det=False,
                use_rec=True,
                use_cls=False,
            )
        else:
            self._fallback_engine = None

    def _initialize_engines(self):
        """Initialize separate Detection and Recognition engines."""
        # Use absolute paths relative to this file (app/core/rapid_ocr_engine.py)
        # Go up 2 levels: app/core -> app -> EasyScanlate
        base_dir = Path(__file__).parent.parent.parent
        
        # 1. Init Detection Engine Only
        self.det_engine = RapidOCR(
            det_model_path=str(base_dir / "OCR/model/ch_PP-OCRv5_mobile_det.onnx"),
            use_det=True,
            use_rec=False,
            use_cls=True,
        )

        # 2. Init Recognition Engine Only (language-specific)
        rec_model_path, rec_keys_path = self._get_rec_model_and_dict(self.language)
        self.rec_engine = RapidOCR(
            rec_model_path=rec_model_path,
            rec_keys_path=rec_keys_path,
            use_det=False,
            use_rec=True,
            use_cls=False,
        )

    def readtext(self, img: np.ndarray) -> List[Tuple[Any, str, float]]:
        """
        Run OCR on an image using the manual Det -> Crop -> Rec pipeline.

        Args:
            img: Input image as numpy array (grayscale or RGB)

        Returns:
            List of tuples: (coordinates, text, confidence)
            Format matches RapidOCR output: ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence)
        """
        results = []

        # Ensure image is in correct format for RapidOCR
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img

        # E7: Downscale very large images — detection accuracy drops above ~2000px wide
        MAX_DET_WIDTH = 2000
        h_orig, w_orig = img_rgb.shape[:2]
        if w_orig > MAX_DET_WIDTH:
            scale = MAX_DET_WIDTH / w_orig
            img_rgb = cv2.resize(img_rgb, (MAX_DET_WIDTH, int(h_orig * scale)), interpolation=cv2.INTER_AREA)

        # E5: Pad image so text touching panel edges is detected
        PAD = 20
        img_padded = cv2.copyMakeBorder(img_rgb, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 1. Run Detection Only
        det_output = self.det_engine(img_padded)

        boxes = []

        # Attempt to extract boxes based on object structure
        if hasattr(det_output, "boxes") and det_output.boxes is not None:
            # Structure found in some wrappers (TextDetOutput.boxes)
            boxes = det_output.boxes
        elif isinstance(det_output, (list, tuple)):
            # Structure: [boxes, scores, elapse] or just [boxes]
            if len(det_output) > 0 and isinstance(det_output[0], (list, np.ndarray)):
                boxes = det_output[0]
            elif len(det_output) > 0 and isinstance(det_output[0], tuple):
                # Maybe (box, score) tuples?
                boxes = [x[0] for x in det_output]
        elif hasattr(det_output, "dt_boxes"):
            # Older RapidOCR versions
            boxes = det_output.dt_boxes

        if boxes is None or len(boxes) == 0:
            return results

        # 2. For each detected box: Crop -> Recognize
        for i, box in enumerate(boxes):
            try:
                # Ensure box is a numpy array or list of points
                if hasattr(box, "box"):  # Handle object wrapper inside list
                    box = box.box

                # E5: shift box coords back by padding offset before cropping original
                box_shifted = np.array(box, dtype=np.float32)
                if hasattr(box_shifted, "tolist"):
                    box_shifted = box_shifted.copy()
                box_shifted = [[p[0] - PAD, p[1] - PAD] for p in (box_shifted.tolist() if hasattr(box_shifted, "tolist") else box_shifted)]

                # 3. Manual Crop from original (unpadded) image
                cropped_img = get_rotate_crop_image(img_rgb, box_shifted)

                # 4. Run Recognition on Crop
                rec_out = self.rec_engine(cropped_img)

                text = ""
                score = 0.0

                def _parse_rec(rec_out):
                    if isinstance(rec_out, tuple):
                        if rec_out[0] and len(rec_out[0]) > 0:
                            return rec_out[0][0]
                    elif hasattr(rec_out, "txts"):
                        if rec_out.txts and len(rec_out.txts) > 0:
                            return rec_out.txts[0], rec_out.scores[0]
                    elif isinstance(rec_out, list) and len(rec_out) > 0:
                        return rec_out[0]
                    return "", 0.0

                parsed = _parse_rec(rec_out)
                if parsed:
                    text, score = parsed

                # E6: retry with inverted crop if confidence is low (handles white-on-black SFX)
                if self.retry_low_confidence and score < 0.5 and cropped_img is not None:
                    inv_crop = cv2.bitwise_not(cropped_img)
                    rec_out_inv = self.rec_engine(inv_crop)
                    parsed_inv = _parse_rec(rec_out_inv)
                    if parsed_inv and parsed_inv[1] > score:
                        text, score = parsed_inv

                # E4: fallback to secondary language model if still low confidence
                if self.retry_low_confidence and score < 0.5 and cropped_img is not None:
                    self._init_fallback_engine()
                    if self._fallback_engine is not None:
                        rec_out_fb = self._fallback_engine(cropped_img)
                        parsed_fb = _parse_rec(rec_out_fb)
                        if parsed_fb and parsed_fb[1] > score:
                            text, score = parsed_fb

                if text:
                    # Convert box to format expected by RapidOCR: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if hasattr(box, "tolist"):
                        box_list = box.tolist()
                    else:
                        box_list = list(box)

                    # Ensure box_list is a list of 4 points
                    if len(box_list) >= 4:
                        # Take only the 4 corner points
                        coords = []
                        for j in range(4):
                            if j < len(box_list):
                                p = box_list[j]
                                if hasattr(p, "tolist"):
                                    p = p.tolist()
                                coords.append([float(p[0]), float(p[1])])

                        results.append((coords, text, float(score)))

            except Exception as inner_e:
                print(f"Error processing box: {inner_e}")
                import traceback
                traceback.print_exc()
                continue

        # E8: Row-banded reading order — group boxes in same horizontal band, sort x within band
        def _reading_order_sort(items, row_tol=30):
            if not items:
                return items
            def center_y(r):
                ys = [p[1] for p in r[0]]
                return (min(ys) + max(ys)) / 2
            sorted_by_y = sorted(items, key=center_y)
            rows, current_row = [], [sorted_by_y[0]]
            for item in sorted_by_y[1:]:
                if abs(center_y(item) - center_y(current_row[-1])) <= row_tol:
                    current_row.append(item)
                else:
                    rows.append(sorted(current_row, key=lambda r: min(p[0] for p in r[0])))
                    current_row = [item]
            rows.append(sorted(current_row, key=lambda r: min(p[0] for p in r[0])))
            return [item for row in rows for item in row]

        return _reading_order_sort(results)
