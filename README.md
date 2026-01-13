# 로컬 RAG 템플릿 (Ollama + Chroma)

## 개요
- LLM: Ollama `gemma3` (로컬)
- Embedding 모델: Ollama `embeddinggemma`
- Vector DB: Chroma (`chromadb`)

이 저장소는 로컬에서 간단히 RAG(검색 후 생성) 흐름을 시도해볼 수 있는 템플릿입니다.

## 설치
1. Ollama 설치 및 모델 준비
   - Ollama 설치: https://ollama.ai (설치 후 데몬/서비스 필요 시 가동)
   - 모델 가져오기(예시):
     ```bash
     ollama pull gemma3
     ollama pull embeddinggemma
     ```

2. 파이썬 패키지 설치
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 사용 방법
1. `data/` 폴더에 `.txt` 또는 `.md` 파일을 넣으세요.
2. `rag_local_ollama.py`의 `ollama_embed`와 `ollama_generate` 함수를 환경(HTTP/CLI)에 맞게 구현하세요.
   - 예: Ollama의 로컬 HTTP 엔드포인트가 있다면 `requests.post`로 호출
   - 또는 `subprocess`로 `ollama` CLI를 호출해 임베딩/생성 결과를 받아올 수 있습니다.
3. 인덱스 빌드 및 쿼리
   ```bash
   python rag_local_ollama.py
   ```

### 디버깅: 엔드포인트 확인
`rag_local_ollama.py`에는 `test_ollama_endpoints()`라는 헬퍼가 있어, 아래와 같이 HTTP 엔드포인트들이 응답하는 형태를 빠르게 확인할 수 있습니다.

```python
from rag_local_ollama import test_ollama_endpoints
test_ollama_endpoints()
```

응답을 확인한 뒤 `ollama_embed`/`ollama_generate`를 해당 형식에 맞게 구현하세요.

### LangChain 사용 옵션 (선택사항)
LangChain을 사용하면 임베딩/검색/검색-생성( Retrieval + LLM )을 더 간단히 구성할 수 있습니다.

- 설치: `pip install langchain`
- 작업 예시:

```python
from rag_local_ollama import build_or_update_index_langchain, query_langchain

# 문서 색인
build_or_update_index_langchain(docs)

# 질의
res = query_langchain("내 문서에 있는 내용을 요약해줘", k=3)
print(res)
```

> 참고: 코드 상단에서 LangChain import는 try/except로 감싸져 있으며, LangChain이 설치되어 있지 않으면 안내 메시지와 함께 오류가 발생합니다. LangChain/Chroma 버전에 따라 일부 시그니처가 다를 수 있으니 문제가 생기면 알려주세요.

## 참고
- Ollama의 정확한 API(엔드포인트와 요청 포맷)는 Ollama 문서를 참고하고, 템플릿의 스텁을 실제 호출형식으로 바꿔주세요.
- Chroma DB는 로컬에 `chroma_db` 폴더(기본)로 저장됩니다.

---
💡 팁: 먼저 `ollama` CLI로 임베딩을 얻는 스크립트를 간단히 작성해보고, 그 호출부를 `ollama_embed`로 옮기면 빠릅니다.