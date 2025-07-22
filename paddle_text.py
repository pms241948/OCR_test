import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import pytesseract
from tqdm import tqdm
import re

INPUT_DIR = "./ex_pdf"
NEW_PADDLE_TXT_DIR = "./outputs/new_paddle_txt"
os.makedirs(NEW_PADDLE_TXT_DIR, exist_ok=True)

# PaddleOCR – 한국어 모델 (최신 버전 설정)
paddle_ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="korean"
)

def merge_ocr_lines_by_yx(ocr_result, y_thresh=20, x_thresh=10):
    """
    PaddleOCR 결과에서 y좌표가 비슷한 라인끼리 한 줄로 합침
    """
    if not ocr_result or not isinstance(ocr_result, list) or not ocr_result[0]:
        return []

    # 각 라인에서 정보 추출
    lines = []
    for line in ocr_result[0]:
        if len(line) < 2:
            continue
            
        box = line[0]  # 바운딩 박스 좌표
        text_info = line[1]  # 텍스트 정보
        
        # 텍스트와 신뢰도 추출
        if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
            text = text_info[0]
            confidence = text_info[1]
        elif isinstance(text_info, str):
            text = text_info
            confidence = 1.0
        else:
            continue
            
        # 신뢰도가 낮은 텍스트는 제외
        if confidence < 0.7 or not text.strip():
            continue
            
        # 바운딩 박스에서 좌표 계산
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        
        x_min = min(x_coords)
        y_center = sum(y_coords) / 4  # 중심 y좌표 사용
        
        lines.append((y_center, x_min, text.strip(), confidence))

    if not lines:
        return []

    # y좌표 기준 정렬
    lines.sort(key=lambda x: (x[0], x[1]))

    # y좌표가 비슷한 라인끼리 그룹화
    merged_lines = []
    current_group = []
    current_y = lines[0][0]
    
    for y, x, text, conf in lines:
        if abs(y - current_y) <= y_thresh:
            # 같은 줄로 판단
            current_group.append((x, text, conf))
        else:
            # 새로운 줄 시작
            if current_group:
                # x좌표 기준으로 정렬하여 한 줄로 합침
                current_group.sort(key=lambda t: t[0])
                line_text = ' '.join([t[1] for t in current_group])
                merged_lines.append(line_text)
            
            current_group = [(x, text, conf)]
            current_y = y
    
    # 마지막 그룹 처리
    if current_group:
        current_group.sort(key=lambda t: t[0])
        line_text = ' '.join([t[1] for t in current_group])
        merged_lines.append(line_text)
    
    return merged_lines

def post_process_korean_text(text_lines):
    """한국어 텍스트 후처리"""
    if not text_lines:
        return []
    
    processed_lines = []
    
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
            
        # 연속된 공백을 하나로 통합
        line = re.sub(r'\s+', ' ', line)
        
        # 한글 조사/어미가 분리된 경우 연결
        particles = ['이', '가', '을', '를', '에', '의', '와', '과', '로', '으로', 
                    '부터', '까지', '에서', '에게', '한테', '께', '라', '아', '야']
        for particle in particles:
            line = re.sub(f'([가-힣])\\s+{particle}\\b', f'\\1{particle}', line)
        
        # 문장 부호 앞 공백 제거
        line = re.sub(r'\s+([.,!?;:])', r'\1', line)
        
        # 괄호 안 공백 정리
        line = re.sub(r'\(\s+', '(', line)
        line = re.sub(r'\s+\)', ')', line)
        
        processed_lines.append(line)
    
    return processed_lines

def merge_short_lines(lines, min_length=10):
    """적절한 줄바꿈을 유지하면서 너무 짧은 줄들만 병합"""
    if not lines:
        return []
    
    merged = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].strip()
        if not current_line:
            i += 1
            continue
        
        # 제목이나 목록 항목 패턴 확인
        is_title_or_list = (
            re.match(r'^[0-9]+\.', current_line) or  # 번호 목록
            re.match(r'^[•\-\*]', current_line) or   # 불릿 목록  
            re.match(r'^[가-힣]\)', current_line) or  # 한글 목록
            re.match(r'^제\s*\d+', current_line) or   # 제N장/조 등
            re.match(r'^[IVX]+\.', current_line) or   # 로마숫자
            (len(current_line) < 20 and current_line.isupper())  # 짧은 대문자 제목
        )
        
        # 제목이나 목록은 그대로 유지
        if is_title_or_list:
            merged.append(current_line)
            i += 1
            continue
        
        # 현재 줄이 너무 짧고 문장이 끝나지 않은 경우에만 다음 줄과 병합 고려
        if (len(current_line) < min_length and 
            not current_line.endswith(('.', '!', '?', '다', '음', '임', '요', '니다', '습니다')) and
            i + 1 < len(lines)):
            
            next_line = lines[i + 1].strip()
            # 다음 줄이 목록이나 제목이 아닌 경우에만 병합
            next_is_title = (
                re.match(r'^[0-9]+\.', next_line) or
                re.match(r'^[•\-\*]', next_line) or
                re.match(r'^[가-힣]\)', next_line) or
                re.match(r'^제\s*\d+', next_line)
            )
            
            if not next_is_title and next_line:
                merged.append(current_line + " " + next_line)
                i += 2  # 다음 줄도 처리했으므로 2 증가
            else:
                merged.append(current_line)
                i += 1
        else:
            # 일반적인 줄은 그대로 유지
            merged.append(current_line)
            i += 1
    
    return merged

def ocr_paddle(pdf_path: str) -> str:
    """개선된 PaddleOCR 텍스트 추출"""
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        all_lines = []
        
        for page_num, pil_img in enumerate(pages):
            try:
                # PIL 이미지를 OpenCV 형식으로 변환
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # OCR 수행
                result = paddle_ocr.ocr(img_bgr)
                
                if result and isinstance(result, list) and len(result) > 0:
                    # 최신 PaddleOCR 구조 확인
                    if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                        # 구버전 형식
                        page_lines = result[0]['rec_texts']
                        all_lines.extend([line.strip() for line in page_lines if line.strip()])
                    else:
                        # 신버전 형식 - 좌표 기반 병합 사용
                        merged_lines = merge_ocr_lines_by_yx(result, y_thresh=20, x_thresh=10)
                        all_lines.extend(merged_lines)
                
                # 페이지 간 구분을 위한 빈 줄 (선택사항)
                if page_num < len(pages) - 1:  # 마지막 페이지가 아닌 경우
                    all_lines.append("")  # 페이지 구분
                    
            except Exception as e:
                print(f"페이지 {page_num + 1} 처리 중 오류: {e}")
                continue
        
        # 텍스트 후처리
        processed_lines = post_process_korean_text(all_lines)
        
        # 빈 줄 제거 (하지만 구조는 유지)
        non_empty_lines = [line for line in processed_lines if line.strip()]
        
        # 너무 짧은 줄만 선별적으로 병합
        merged_lines = merge_short_lines(non_empty_lines, min_length=10)
        
        # 최종 텍스트 정리
        final_text = "\n".join(merged_lines)
        
        # 과도한 연속 줄바꿈만 정리 (기본 줄바꿈은 유지)
        final_text = re.sub(r'\n{4,}', '\n\n\n', final_text)
        
        return final_text.strip()
        
    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({pdf_path}): {e}")
        return ""

def process_pdfs():
    """PDF 파일들을 일괄 처리"""
    if not os.path.exists(INPUT_DIR):
        print(f"입력 디렉토리가 존재하지 않습니다: {INPUT_DIR}")
        return
    
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"PDF 파일이 없습니다: {INPUT_DIR}")
        return
    
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
    
    success_count = 0
    error_count = 0
    
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            stem = os.path.splitext(pdf)[0]
            pdf_path = os.path.join(INPUT_DIR, pdf)
            
            # PaddleOCR로 텍스트 추출
            extracted_text = ocr_paddle(pdf_path)
            
            if extracted_text:
                # 결과 파일 저장
                output_path = os.path.join(NEW_PADDLE_TXT_DIR, f"{stem}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                success_count += 1
            else:
                print(f"텍스트 추출 실패: {pdf}")
                error_count += 1
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({pdf}): {e}")
            error_count += 1
    
    print(f"\n처리 완료: 성공 {success_count}개, 실패 {error_count}개")

# 실행
if __name__ == "__main__":
    process_pdfs()