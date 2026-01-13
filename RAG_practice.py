import os
import sys

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

PDFS_DIR = "pdfs"
VSTORE_DIR = "vectorstore"

def load_and_split(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_vectorstore(batch_size: int = 100):
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)

    total_indexed = 0
    pdfs = list(Path(PDFS_DIR).glob("*.pdf"))

    if not pdfs:
        print("No PDF files found in:", PDFS_DIR)
        return
    
    for p in pdfs:
        print("Processing:", p)
        docs = load_and_split(p)
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            if not batch:
                continue
            db.add_documents(batch)
            total_indexed += len(batch)
            print(f"  Indexed {total_indexed} documents so far")
            # free memory immediately after persisting
            del batch
            import gc
            gc.collect()
        del docs
        import gc
        gc.collect()
    
    print("Vectorstore built and persisted at:", VSTORE_DIR)

def get_embeddings():
    emb = OllamaEmbeddings(model="embeddinggemma")
    return emb

def query_loop(question: str):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOllama(model="gemma3", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = qa_chain.invoke(question)
    print("Answer:", answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RAG_practice.py [build | query \"your question\"]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build":
        build_vectorstore()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Provide a question: python RAG_practice.py query \"질문\"")
            sys.exit(1)
        query_loop(sys.argv[2])
    else:
        print("Unknown command:", cmd)