# 로컬 RAG 템플릿 (Ollama + Chroma)

## 개요
- LLM: Ollama `gemma3` (로컬)
- Embedding 모델: Ollama `embeddinggemma`
- Vector DB: Chroma (`chromadb`)

이 저장소는 로컬에서 간단히 RAG(검색 후 생성) 흐름을 시도해볼 수 있는 템플릿입니다.

## Ollama 설치 및 모델 준비
1. Ollama 설치: https://ollama.ai (설치 후 데몬/서비스 필요 시 가동)
2. 모델 가져오기:
     ```bash
     ollama pull gemma3 (or llama3.1)
     ollama pull embeddinggemma (or qwen3-embedding)
     ```

2. 파이썬 패키지 설치
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 사용 방법
1. `data/` 폴더에 학습시킬 자료를 `.pdf` 형태의 파일로 넣으세요.
2. 최초 사용 전, 벡터 저장소를 리셋하고 다시 파일을 불러옵니다.
   ```bash
   python RAGbuilder.py reset
   python RAGbuilder.py build
   ```
3. 학습된 문서의 리스트는 `check`로 확인할 수 있습니다.
   ```bash
   python RAGbuilder.py check
   ```
4. 질의를 위해 streamlit으로 챗봇을 시작해주세요.
   ```bash
   streamlit run app.py
   ```