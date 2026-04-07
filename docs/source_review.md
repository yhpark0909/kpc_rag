# 소스 분석 리뷰

## 핵심 이슈

1. `app.py`가 `RAGbuilder_improved`를 import하고 있으나 실제 저장소에는 `RAGbuilder.py`만 존재합니다.
   - 현재 상태로는 앱 실행 시 ImportError 가능성이 높습니다.

2. `README.md`는 `streamlit run app.py`를 안내하지만, `app.py`는 `chainlit` 기반입니다.
   - 실행 가이드와 앱 프레임워크가 불일치합니다.

3. `RAGbuilder.py`는 `.pdf`와 `.md`를 모두 색인하도록 개선되어 있으며, Markdown 헤더 기반 분할 + 청크 분할을 적용한 점은 좋습니다.

4. 데이터 폴더의 규정 파일 포맷은 `# 제목` + `## 장` + `### 절` + `#### 조` 형태로 일관되어 있으며 RAG 청킹에 유리합니다.

## 권장 조치

- `app.py` import 경로 정리 (`RAGbuilder` 또는 실제 개선 모듈 파일 추가).
- `README.md` 실행 가이드를 `chainlit run app.py`로 수정하거나 앱을 streamlit 기반으로 통일.
- PDF→MD 변환 파이프라인을 스크립트로 표준화해 신규 규정 투입 시 수작업을 최소화.
