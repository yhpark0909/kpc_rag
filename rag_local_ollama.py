"""
간단한 로컬 RAG 템플릿
- LLM / 임베딩: Ollama (gemma3, embeddinggemma) — 구현부에 맞춰 HTTP/CLI 호출로 연결하세요
- 벡터 저장소: Chroma (chromadb)

파일 역할:
- load_documents: 로컬 텍스트를 읽어 chunking
- embed_texts: Ollama embedding 호출의 래퍼(구현 필요)
- build_or_update_index: 문서 임베딩을 Chroma에 업서트
- query: 쿼리 임베딩을 얻어 top-k 문서 검색 후 LLM으로 응답 생성

주의: Ollama의 정확한 API(HTTP 경로 또는 CLI)를 사용자의 환경에 맞게 구현해야 합니다. 아래는 구조적 템플릿입니다.
"""

import os
import json
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

# 외부 호출에 사용할 모듈 (구현 시 활성화)
import requests
# LangChain (옵션): 설치되어 있으면 통합 기능을 제공합니다.
try:
    from langchain.embeddings.base import Embeddings
    from langchain.llms.base import LLM
    from langchain.vectorstores import Chroma as LCChroma
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except Exception:
    Embeddings = None  # type: ignore
    LLM = None
    LCChroma = None
    RetrievalQA = None
    Document = None

# --- 설정 ---
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # 환경에 맞게 수정
EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "gemma3"

# --- 유틸리티 ---

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """간단한 텍스트 청킹(토크나이저 기반이 아니므로 간단히 자름)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


def load_documents_from_folder(folder: str) -> List[Dict[str, Any]]:
    """폴더 내 .txt/.md 파일을 읽어 문서 목록 반환
    반환 항목: {'id': id, 'text': text, 'meta': {...}}
    """
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, fname)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append({
                'id': os.path.relpath(path, folder),
                'text': text,
                'meta': {'source': path}
            })
    return docs


# --- Ollama 연동 스텁(사용자 환경에 맞게 구현하세요) ---

def ollama_embed(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Ollama로부터 임베딩을 얻는 함수(간단한 HTTP 시도 구현)

    주: Ollama의 실제 엔드포인트/응답 형식은 환경에 따라 다를 수 있습니다.
    아래는 여러 흔한 경로를 시도해보고, 공통적인 응답 포맷(OpenAI-like 등)을 파싱합니다.
    필요하면 CLI(subprocess) 방식으로 대체하세요.
    """
    endpoints = [
        f"{OLLAMA_HOST}/api/embeddings",
        f"{OLLAMA_HOST}/api/embed",
        f"{OLLAMA_HOST}/embed",
        f"{OLLAMA_HOST}/v1/embeddings",
    ]
    payload = {"model": model, "input": texts}
    headers = {"Content-Type": "application/json"}
    last_exc = None
    for url in endpoints:
        try:
            resp = requests.post(url, json=payload, timeout=30, headers=headers)
            if not resp.ok:
                continue
            j = resp.json()
            # OpenAI-like: {"data": [{"embedding": [...]}]}
            if isinstance(j, dict) and 'data' in j:
                data = j['data']
                if isinstance(data, list) and data and 'embedding' in data[0]:
                    return [item['embedding'] for item in data]
            # simple list of embeddings: [[...],[...]]
            if isinstance(j, list) and j and isinstance(j[0], list):
                return j
            # {"embeddings": [...]}
            if isinstance(j, dict) and 'embeddings' in j:
                return j['embeddings']
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"임베딩을 얻지 못했습니다. Ollama 엔드포인트/CLI를 확인하고 `ollama_embed`를 환경에 맞게 구현하세요. (마지막 오류: {last_exc})")


def ollama_generate(prompt: str, model: str = LLM_MODEL, max_tokens: int = 512) -> str:
    """LLM(예: gemma3)을 호출해 텍스트 생성(간단한 HTTP 시도).

    여러 엔드포인트를 시도하며, 공통적인 응답 포맷을 파싱합니다. 환경에 맞지 않으면 CLI 방식으로 대체하세요.
    """
    endpoints = [
        f"{OLLAMA_HOST}/api/generate",
        f"{OLLAMA_HOST}/generate",
        f"{OLLAMA_HOST}/v1/generate",
        f"{OLLAMA_HOST}/api/chat",
    ]
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    last_exc = None
    for url in endpoints:
        try:
            resp = requests.post(url, json=payload, timeout=60, headers=headers)
            if not resp.ok:
                continue
            j = resp.json()
            # OpenAI-like: {'choices':[{'message':{'content': '...'}}]}
            if isinstance(j, dict):
                if 'choices' in j and isinstance(j['choices'], list) and j['choices']:
                    ch = j['choices'][0]
                    if isinstance(ch, dict):
                        if 'message' in ch and isinstance(ch['message'], dict) and 'content' in ch['message']:
                            return ch['message']['content']
                        if 'text' in ch:
                            return ch['text']
                if 'output' in j and isinstance(j['output'], str):
                    return j['output']
                if 'text' in j and isinstance(j['text'], str):
                    return j['text']
            if isinstance(j, str):
                return j
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"LLM 응답을 얻지 못했습니다. Ollama 엔드포인트/CLI를 확인하고 `ollama_generate`를 환경에 맞게 구현하세요. (마지막 오류: {last_exc})")


def test_ollama_endpoints():
    """간단한 헬퍼: 여러 엔드포인트에 샘플 요청을 보내보고 응답 형태를 출력합니다.

    이 함수로 어떤 경로/응답이 돌아오는지 확인한 뒤 `ollama_embed`/`ollama_generate`를 맞춰 구현하세요.
    """
    tests = [
        ("/api/embeddings", {"model": EMBEDDING_MODEL, "input": ["hello world"]}),
        ("/api/embed", {"model": EMBEDDING_MODEL, "input": ["hello world"]}),
        ("/api/generate", {"model": LLM_MODEL, "prompt": "Hello", "max_tokens": 32}),
        ("/generate", {"model": LLM_MODEL, "prompt": "Hello", "max_tokens": 32}),
    ]
    for path, payload in tests:
        url = OLLAMA_HOST.rstrip("/") + path
        try:
            r = requests.post(url, json=payload, timeout=10)
            print(f"[{url}] status={r.status_code} text_sample={r.text[:400]}")
        except Exception as e:
            print(f"[{url}] error: {e}")

# --- LangChain 통합 (옵션) ---
# 아래 클래스를 통해 LangChain의 Embeddings/LLM 인터페이스를 구현합니다.
# LangChain이 설치되어 있지 않다면 위에서 임포트가 실패하고 None으로 설정됩니다. 설치한 뒤 사용하세요.

class OllamaEmbeddings(Embeddings):
    """LangChain Embeddings 인터페이스를 Ollama 임베딩으로 구현"""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return ollama_embed(texts, model=self.model)

    def embed_query(self, text: str) -> List[float]:
        return ollama_embed([text], model=self.model)[0]


class OllamaLLM(LLM):
    """LangChain LLM 인터페이스를 Ollama 생성기로 구현"""
    def __init__(self, model: str = LLM_MODEL, max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:  # for LangChain metadata
        return "ollama"

    def _call(self, prompt: str, stop: List[str] | None = None) -> str:
        return ollama_generate(prompt, model=self.model, max_tokens=self.max_tokens)

    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}


def build_or_update_index_langchain(documents: List[Dict[str, Any]], collection_name: str = "rag_collection"):
    """LangChain을 활용한 인덱스 생성/업데이트 헬퍼

    내부적으로 `OllamaEmbeddings`를 사용해 임베딩을 생성하고 `langchain.vectorstores.Chroma`로 저장합니다.
    """
    emb = OllamaEmbeddings()
    texts = []
    metadatas = []
    ids = []
    for doc in documents:
        chunks = chunk_text(doc['text'])
        for i, c in enumerate(chunks):
            texts.append(c)
            md = dict(doc['meta'])
            md.update({'chunk': i})
            metadatas.append(md)
            ids.append(f"{doc['id']}_chunk_{i}")

    # LangChain Chroma의 시그니처 차이를 감안해 두 가지 방식 시도
    try:
        vect = LCChroma.from_texts(texts, embeddings=emb, metadatas=metadatas, ids=ids, persist_directory=CHROMA_DIR, collection_name=collection_name)
    except TypeError:
        vect = LCChroma.from_texts(texts, embedding=emb, metadatas=metadatas, ids=ids, persist_directory=CHROMA_DIR, collection_name=collection_name)

    return vect


def query_langchain(query_text: str, collection_name: str = "rag_collection", k: int = 4) -> str:
    """LangChain 기반 검색+생성

    - Chroma 컬렉션을 로드하고 retriever를 생성한 뒤 `RetrievalQA` 체인을 실행합니다.
    - LangChain/Chroma 버전이 다르면 로드에 실패할 수 있으므로 오류를 명확히 안내합니다.
    """
    emb = OllamaEmbeddings()
    llm = OllamaLLM()
    try:
        vect = LCChroma(persist_directory=CHROMA_DIR, collection_name=collection_name, embedding_function=emb)
        retriever = vect.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise RuntimeError("LangChain의 Chroma 로드에 실패했습니다. LangChain/Chroma 버전과 설정을 확인하세요. 오류: " + str(e))

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# --- Chroma 연동 ---


# --- Chroma 연동 ---

def get_chroma_client(persist_directory: str = CHROMA_DIR):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client


def build_or_update_index(documents: List[Dict[str, Any]], collection_name: str = "rag_collection"):
    """문서를 청킹하고 임베딩을 계산해 Chroma에 업서트한다."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    ids = []
    documents_texts = []
    metadatas = []

    for doc in documents:
        chunks = chunk_text(doc['text'])
        for i, c in enumerate(chunks):
            uid = f"{doc['id']}_chunk_{i}"
            ids.append(uid)
            documents_texts.append(c)
            md = dict(doc['meta'])
            md.update({'chunk': i})
            metadatas.append(md)

    # 임베딩 얻기
    embeddings = ollama_embed(documents_texts)

    # 업서트
    collection.upsert(
        ids=ids,
        documents=documents_texts,
        metadatas=metadatas,
        embeddings=embeddings
    )
    # 영속화
    client.persist()
    return collection


def query(query_text: str, collection_name: str = "rag_collection", k: int = 4) -> str:
    """쿼리 -> 검색 -> LLM과 결합해 응답 생성"""
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    q_emb = ollama_embed([query_text])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])

    retrieved_docs = []
    for docs_list in res['documents']:
        # res['documents']는 리스트의 리스트
        retrieved_docs.extend(docs_list)

    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"You are a helpful assistant. Use the following context to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query_text}\n\nAnswer concisely."

    answer = ollama_generate(prompt)
    return answer


# --- 간단 데모 ---
if __name__ == "__main__":
    # 예시: data 폴더의 텍스트를 색인하고 간단 쿼리 실행
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    docs = load_documents_from_folder(data_dir)
    if not docs:
        print("data 폴더에 .txt 또는 .md 파일을 넣어주세요. (예: data/example.txt)")
        exit(1)

    print(f"Loaded {len(docs)} documents. Building index (구현된 ollama 함수 필요)...")
    try:
        build_or_update_index(docs)
    except NotImplementedError as e:
        print(e)
        print("ollama 관련 함수(임베딩/생성)를 구현한 후 다시 시도하세요.")
        exit(1)

    q = input("질문을 입력하세요: ")
    print("검색 중...")
    try:
        ans = query(q)
        print("Answer:\n", ans)
    except NotImplementedError as e:
        print(e)
        print("ollama 관련 함수(임베딩/생성)를 구현한 후 다시 시도하세요.")
```