import sys
import gc
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma

# 설정값
DOCS_DIR = "data"
VSTORE_DIR = "vectorstore"
EMBEDDINGMODEL_PATH = "./models/Qwen3-Embedding-4B-OV"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_split(pdf_path: Path) -> List:
    """PDF 파일을 로드하고 청킹"""
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            is_separator_regex=True,
            separators=[r"\n\n", r"(?=제\d+조\()", r"\n", r"\.", r"\s+"]
        )
        return splitter.split_documents(docs)
    
    except Exception as e:
        logger.error(f"PDF 로드 실패 {pdf_path.name}: {e}")
        return []


def load_and_split_md(file_path: Path) -> List:
    """마크다운 파일을 로드하고 청킹 (헤더 기반 + 크기 기반)"""
    try:
        loader = TextLoader(str(file_path), encoding='utf-8')
        data = loader.load()
        content = data[0].page_content

        headers_to_split_on = [
            ("#", "Header 1"), ("##", "Header 2"), 
            ("###", "Header 3"), ("####", "Header 4"),
        ]

        # 1차 분할: 헤더 기준
        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        md_header_splits = md_header_splitter.split_text(content)

        # 메타데이터에 source 추가
        for split in md_header_splits:
            split.metadata["source"] = str(file_path)

        # 2차 분할: 크기 기준
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            is_separator_regex=True,
            separators=[r"\n\n", r"(?=제\d+조\()", r"\n", r"\.", r"\s+", ""]
        )
        
        return text_splitter.split_documents(md_header_splits)
    
    except Exception as e:
        logger.error(f"마크다운 로드 실패 {file_path.name}: {e}")
        return []


def get_embeddings(device: str = "CPU") -> Optional[OpenVINOEmbeddings]:
    """
    임베딩 모델 로드 (16GB RAM 환경 최적화)
    
    Args:
        device: "CPU" 권장 (Intel iGPU와 RAM 공유 방지)
    
    Returns:
        OpenVINOEmbeddings 인스턴스 또는 None (오류 시)
    """
    try:
        emb = OpenVINOEmbeddings(
            model_name_or_path=EMBEDDINGMODEL_PATH,
            model_kwargs={
                "device": device,
                "compile": True,
                "ov_config": {
                    "CACHE_DIR": "./ov_cache",
                    },
                "fix_mistral_regex": True,
                "trust_remote_code": True,
            }
        )
        logger.info(f"임베딩 모델 로드 완료 (device: {device}, 메모리 절약 모드)")
        return emb
    
    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        return None


def build_vectorstore(batch_size: int = 2, device: str = "CPU") -> bool:
    """
    벡터 저장소 구축
    
    Args:
        batch_size: 배치 크기 (낮은 사양에서는 2 권장)
        device: 임베딩 모델 디바이스 ("CPU" 권장)
    
    Returns:
        성공 여부
    """
    logger.info("벡터 저장소 구축 시작...")
    
    # 임베딩 모델 로드
    embeddings = get_embeddings(device=device)
    if embeddings is None:
        return False
    
    try:
        db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
        
        # 지원 파일 탐색
        data_path = Path(DOCS_DIR)
        if not data_path.exists():
            logger.error(f"데이터 디렉토리가 없습니다: {DOCS_DIR}")
            return False
        
        data_files = [
            p for p in data_path.iterdir() 
            if p.suffix.lower() in [".md", ".pdf"] and p.is_file()
        ]

        if not data_files:
            logger.warning(f"지원되는 파일(.md, .pdf)이 없습니다: {DOCS_DIR}")
            return False
        
        total_indexed = 0
        
        for file_path in data_files:
            logger.info(f"처리 중: {file_path.name}")
            
            # 파일별 문서 로드
            if file_path.suffix.lower() == ".pdf":
                docs = load_and_split(file_path)
            else:  # .md
                docs = load_and_split_md(file_path)
            
            if not docs:
                logger.warning(f"문서를 로드할 수 없습니다: {file_path.name}")
                continue
            
            # 배치별 인덱싱 (메모리 절약)
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                try:
                    db.add_documents(batch)
                    total_indexed += len(batch)
                    logger.info(f"  인덱싱 완료: {total_indexed} 청크")
                except Exception as e:
                    logger.error(f"배치 인덱싱 실패: {e}")
                
                # 메모리 정리
                del batch
                gc.collect()
            
            del docs
            gc.collect()
        
        # 임베딩 모델 메모리 해제
        del embeddings
        gc.collect()
        
        logger.info(f"✅ 벡터 저장소 구축 완료: {VSTORE_DIR}")
        logger.info(f"총 {total_indexed}개 청크 인덱싱됨")
        return True
    
    except Exception as e:
        logger.error(f"벡터 저장소 구축 실패: {e}")
        return False


def check_vectorstore_contents(device: str = "CPU") -> bool:
    """벡터 저장소 내용 확인"""
    embeddings = get_embeddings(device=device)
    if embeddings is None:
        return False
    
    try:
        db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
        all_docs = db.get()
        
        if not all_docs or not all_docs['documents']:
            logger.info("벡터 저장소가 비어있습니다.")
            return True
        
        # 소스별 청크 수 집계
        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'Unknown')
            file_name = Path(source).name
            sources[file_name] = sources.get(file_name, 0) + 1
        
        print(f"\n=== 벡터 저장소 현황 ===")
        print(f"총 임베딩 청크: {len(all_docs['documents'])}개")
        print(f"저장 위치: {VSTORE_DIR}")
        print(f"\n파일별 청크 수:")
        for file_name, count in sorted(sources.items()):
            print(f"  - {file_name}: {count}개")
        print()
        
        return True
    
    except Exception as e:
        logger.error(f"벡터 저장소 확인 실패: {e}")
        return False


def reset_vectorstore() -> bool:
    """벡터 저장소 초기화"""
    import shutil
    
    vstore_path = Path(VSTORE_DIR)
    
    if not vstore_path.exists():
        logger.info(f"벡터 저장소가 존재하지 않습니다: {VSTORE_DIR}")
        return True
    
    try:
        shutil.rmtree(vstore_path)
        logger.info(f"✅ 벡터 저장소 초기화 완료: {VSTORE_DIR}")
        return True
    
    except Exception as e:
        logger.error(f"벡터 저장소 초기화 실패: {e}")
        return False


def validate_setup() -> bool:
    """환경 설정 검증"""
    issues = []
    
    # 데이터 디렉토리 확인
    if not Path(DOCS_DIR).exists():
        issues.append(f"데이터 디렉토리 없음: {DOCS_DIR}")
    
    # 임베딩 모델 확인
    if not Path(EMBEDDINGMODEL_PATH).exists():
        issues.append(f"임베딩 모델 없음: {EMBEDDINGMODEL_PATH}")
    
    # OpenVINO 캐시 디렉토리 확인/생성
    cache_dir = Path("./ov_cache")
    cache_dir.mkdir(exist_ok=True)
    
    if issues:
        logger.error("환경 설정 문제:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("환경 설정 검증 완료")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python RAGbuilder_improved.py [build | check | reset | validate]")
        print("  build    - 벡터 저장소 구축")
        print("  check    - 벡터 저장소 내용 확인")
        print("  reset    - 벡터 저장소 초기화")
        print("  validate - 환경 설정 검증")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "build":
        success = build_vectorstore()
        sys.exit(0 if success else 1)
    
    elif cmd == "check":
        success = check_vectorstore_contents()
        sys.exit(0 if success else 1)
    
    elif cmd == "reset":
        success = reset_vectorstore()
        sys.exit(0 if success else 1)
    
    elif cmd == "validate":
        success = validate_setup()
        sys.exit(0 if success else 1)
    
    else:
        logger.error(f"알 수 없는 명령: {cmd}")
        sys.exit(1)
