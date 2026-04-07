# PDF → Markdown 변환 가이드

## 1) 준비
아래 패키지를 먼저 설치하세요.

```bash
pip install langchain-community pymupdf
```

> 프로젝트 환경을 그대로 쓰려면 `pip install -r requirements.txt` 후 위 명령을 추가로 실행하세요.

## 2) 단일 PDF 변환
`data/(4-11) 여비지급규칙_260130.pdf`를 같은 폴더에 `.md`로 변환:

```bash
python tools/convert_pdf_to_md.py "data/(4-11) 여비지급규칙_260130.pdf"
```

출력 파일: `data/(4-11) 여비지급규칙_260130.md`

## 3) data 폴더 전체 PDF 일괄 변환

```bash
python tools/convert_pdf_to_md.py data
```

`data` 안의 모든 `.pdf`를 같은 이름의 `.md`로 변환합니다.

## 4) 변환 규칙
스크립트는 텍스트를 추출한 뒤 아래 규칙으로 헤더를 정규화합니다.

- `제N장` → `##`
- `제N절` → `###`
- `제N조(` → `####`

나머지 줄은 본문으로 유지합니다.

## 5) 문제 해결
- `pymupdf package not found` 오류: `pip install pymupdf` 실행
- 추출 텍스트가 비어 있음: 스캔본 PDF일 가능성. OCR 후 재변환 필요
