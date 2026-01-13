import os
import sys

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

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

if __name__ == "__main__":
    