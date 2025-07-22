# PDF OCR 성능 비교 프로젝트 (PaddleOCR & Tesseract)

## 개요

이 프로젝트는 PDF 파일에서 텍스트를 추출하기 위해 **PaddleOCR**와 **Tesseract** 두 가지 엔진을 사용하여 성능을 비교하는 실험 코드입니다. 각 PDF 파일에 대해 두 엔진의 결과를 각각 txt 파일로 저장합니다.

---

## 문제 상황 및 해결 과정

- **문제:**  
  PaddleOCR의 최신 버전에서 결과가 한 글자씩만 추출되거나, 의미 없는 값이 나오는 현상 발생  
  (예: `n`, `a`, `o`, ...)

- **원인:**  
  PaddleOCR의 결과 반환 구조가 변경되었으나, 기존 코드가 예전 방식(`for line in result[0]: ...`)으로 파싱함

- **해결:**  
  최신 구조에 맞게 `result[0]['rec_texts']`에서 텍스트를 추출하거나,  
  **좌표(y, x) 기반 병합, 신뢰도 필터링, 한글 후처리, 레이아웃 보존 등 다양한 후처리(post-processing) 기법을 적용**하여 가독성과 문서 구조를 크게 개선함

---

## Conda 환경 구성

아래 환경에서 정상 동작을 확인했습니다.

### 1. Conda 환경 생성

```bash
conda create -n ocr_test python=3.10
conda activate ocr_test
```

### 2. 필수 패키지 설치

```bash
pip install paddlepaddle==2.5.2
pip install paddleocr==2.7.1.3
pip install pytesseract
pip install opencv-python
pip install pdf2image
pip install tqdm
pip install numpy
```

> **참고:**  
> - Windows의 경우, [PaddlePaddle 공식 설치 가이드](https://www.paddlepaddle.org.cn/install/quick) 참고  
> - GPU 사용 시, CUDA 버전에 맞는 paddlepaddle-gpu 설치 필요

---

## 시스템 환경변수 및 외부 프로그램

### 1. Tesseract 설치

- [Tesseract 공식 설치 페이지](https://github.com/tesseract-ocr/tesseract)에서 Windows용 설치 파일 다운로드 및 설치
- 설치 후, 시스템 환경변수에 Tesseract 경로 추가  
  (예: `C:\Program Files\Tesseract-OCR`)
- 또는 코드에서 직접 경로 지정:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### 2. PDF to Image 변환을 위한 Poppler 설치

- [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)에서 다운로드
- 압축 해제 후, bin 폴더 경로를 시스템 환경변수 `PATH`에 추가  
  (예: `C:\tools\poppler-xx\bin`)

---

## 실행 방법

1. PDF 파일을 `ex_pdf/` 폴더에 넣습니다.
2. `paddle_text.py`(최신 개선 버전) 또는 Jupyter Notebook(`ocr_test.ipynb`)을 실행합니다.
3. 모든 셀 또는 스크립트를 실행하면,  
   - `outputs/new_paddle_txt/`  
   - `outputs/tesseract_txt/`  
   폴더에 결과가 저장됩니다.

---

## 주요 코드(핵심 부분, 최신 개선 버전)

```python
def merge_ocr_lines_by_yx(ocr_result, y_thresh=20, x_thresh=10):
    # ... (좌표 기반 병합 함수, paddle_text.py 참고)
    # y좌표가 비슷한 라인끼리 한 줄로 합침
    # ...
    return merged_lines

def post_process_korean_text(text_lines):
    # ... (한글 조사/어미/공백/문장부호 후처리)
    return processed_lines

def merge_short_lines(lines, min_length=10):
    # ... (짧은 줄 병합, 제목/목록/표 구분)
    return merged

def ocr_paddle(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    all_lines = []
    for page_num, pil_img in enumerate(pages):
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result = paddle_ocr.ocr(img_bgr)
        if result and isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                all_lines.extend([line.strip() for line in result[0]['rec_texts'] if line.strip()])
            else:
                merged_lines = merge_ocr_lines_by_yx(result, y_thresh=20, x_thresh=10)
                all_lines.extend(merged_lines)
        if page_num < len(pages) - 1:
            all_lines.append("")
    processed_lines = post_process_korean_text(all_lines)
    non_empty_lines = [line for line in processed_lines if line.strip()]
    merged_lines = merge_short_lines(non_empty_lines, min_length=10)
    final_text = "\n".join(merged_lines)
    final_text = re.sub(r'\n{4,}', '\n\n\n', final_text)
    return final_text.strip()
```

---

## 자주 발생하는 이슈 및 해결 팁

- **PaddleOCR 결과가 한 글자씩만 나오는 경우:**  
  → PaddleOCR 결과 구조에 따라 `result[0]['rec_texts']` 또는 좌표 기반 병합/후처리 함수 사용

- **줄바꿈이 너무 많거나, 문장/문단이 어색하게 쪼개지는 경우:**  
  → 좌표(y, x) 기반 병합, 한글 후처리, 짧은 줄 병합 등 최신 paddle_text.py 방식 적용

- **Tesseract가 동작하지 않는 경우:**  
  → 환경변수 또는 코드에서 Tesseract 경로를 명확히 지정

- **pdf2image 오류:**  
  → Poppler 설치 및 환경변수 등록 필요

- **PaddleOCR 모델 자동 다운로드 실패:**  
  → 인터넷 연결 확인, 또는 `.paddlex/official_models` 폴더 삭제 후 재실행

---

## 참고

- [PaddleOCR 공식 문서](https://github.com/PaddlePaddle/PaddleOCR)
- [Tesseract 공식 문서](https://github.com/tesseract-ocr/tesseract)
- [pdf2image 공식 문서](https://pypi.org/project/pdf2image/)

---

## 폴더 구조 예시

```
OCR_test/
  ex_pdf/                # 입력 PDF 파일
  outputs/
    new_paddle_txt/      # PaddleOCR 최신 개선 결과
    tesseract_txt/       # Tesseract 결과
  paddle_text.py         # PaddleOCR 개선 버전 메인 스크립트
  ocr_test.ipynb         # (선택) Jupyter 노트북
  models/                # (선택) 커스텀 모델
  pdf_original/          # (선택) 원본 PDF
```

---

## 문의

문제 발생 시,  
- 에러 메시지  
- 환경 정보  
- 샘플 PDF  
를 함께 첨부해주시면 빠른 해결이 가능합니다. 