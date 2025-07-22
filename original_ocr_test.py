import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import pytesseract
from tqdm import tqdm

INPUT_DIR = "./ex_pdf"
PADDLE_TXT_DIR = "./outputs/paddle_txt"
TESSERACT_TXT_DIR = "./outputs/tesseract_txt"
os.makedirs(PADDLE_TXT_DIR, exist_ok=True)
os.makedirs(TESSERACT_TXT_DIR, exist_ok=True)

# PaddleOCR – 한국어 모델
paddle_ocr = PaddleOCR(
        use_textline_orientation=True,  # 최신 옵션 사용
        lang="korean"
)
# Tesseract – 한·영 동시
# 윈도우 PATH 지정 필요 시: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "kor+eng"
TESS_CONFIG = "--psm 6"

def ocr_paddle(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    lines = []
    for pil_img in pages:
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result = paddle_ocr.ocr(img_bgr)
        # 최신 PaddleOCR 구조에 맞게 텍스트 추출
        if result and isinstance(result, list) and 'rec_texts' in result[0]:
            lines.extend(result[0]['rec_texts'])
    return "\n".join(lines)



def ocr_tesseract(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    lines = []
    for pil_img in pages:
        text = pytesseract.image_to_string(pil_img, lang=TESS_LANG, config=TESS_CONFIG)
        lines.append(text)
    return "\n".join(lines)

pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

for pdf in tqdm(pdf_files, desc="Processing PDFs"):
    stem = os.path.splitext(pdf)[0]
    path = os.path.join(INPUT_DIR, pdf)

    # PaddleOCR
    paddle_txt = ocr_paddle(path)
    with open(os.path.join(PADDLE_TXT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(paddle_txt)

    # Tesseract
    tess_txt = ocr_tesseract(path)
    with open(os.path.join(TESSERACT_TXT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(tess_txt)