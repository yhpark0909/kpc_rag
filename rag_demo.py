# rag_demo.py
"""
간단한 RAG 데모:
- Build: PDFs -> chunks -> embeddings -> Chroma에 저장
- Query: 검색 -> LLM(gemma3)로 답변 생성
"""

import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA

PDFS_DIR = Path("pdfs")
VSTORE_DIR = "vectorstore"

def load_and_split(pdf_path: Path) -> List:
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_vectorstore(batch_size: int = 100):
    """Index PDFs incrementally to reduce peak RAM usage.

    - Processes each PDF independently
    - Adds documents in batches to Chroma and persists after each batch
    - Forces garbage collection between batches
    """
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)

    total_indexed = 0
    pdfs = list(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in:", PDFS_DIR)
        return

    for p in pdfs:
        print("Processing:", p)
        docs = load_and_split(p)
        # add in smaller batches to avoid holding everything in memory
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            if not batch:
                continue
            db.add_documents(batch)
            #db.persist()
            total_indexed += len(batch)
            print(f"  Indexed {total_indexed} documents so far")
            # free memory immediately after persisting
            del batch
            import gc
            gc.collect()
        # done with this PDF
        del docs
        import gc
        gc.collect()

    print("Vector store created at:", VSTORE_DIR)

def query_loop(q: str):
    # 재사용 가능한 벡터 DB 로딩
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    chat = ChatOllama(model="gemma3", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, return_source_documents=True)
    # Use `invoke` to avoid LangChainDeprecationWarning; fall back to callable for older versions
    try:
        res = qa.invoke({"query": q})
    except AttributeError:
        res = qa({"query": q})
    print("\n=== ANSWER ===")
    print(res["result"])
    print("\n=== SOURCES ===")
    for d in res.get("source_documents", []):
        print("-", d.metadata.get("source"))

def get_embeddings():
    """Try Ollama embeddings first; if unavailable or the model doesn't support embeddings,
    fall back to HuggingFace (sentence-transformers).
    """
    # 1) Try OllamaEmbeddings
    try:
        emb = OllamaEmbeddings(model="embeddinggemma", temperature=0)
        # quick sanity check
        try:
            emb.embed_documents(["test"])
            print("Using OllamaEmbeddings (model supports embeddings).")
            return emb
        except Exception as e:
            print("OllamaEmbeddings initialized but embed call failed:", e)
    except Exception as e:
        print("OllamaEmbeddings initialization failed:", e)

    # 2) Fallback to HuggingFaceEmbeddings
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            emb.embed_documents(["test"])
            print("Falling back to HuggingFaceEmbeddings(all-MiniLM-L6-v2).")
            return emb
        except Exception as e:
            print("HuggingFaceEmbeddings test call failed:", e)
            raise
    except Exception as e:
        raise RuntimeError("No embedding backend available. Install sentence-transformers or provide an Ollama embedding model.") from e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_demo.py [build | query \"your question\"]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build":
        build_vectorstore()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Provide a question: python rag_demo.py query \"질문\"")
            sys.exit(1)
        query_loop(sys.argv[2])
    else:
        print("Unknown command:", cmd)