import chainlit as cl
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 현재 RAGbuilder.py 기준
from RAGbuilder import VSTORE_DIR, get_embeddings

CONTACTS_FILE = Path("contacts") / "rule_contacts.json"
RELEVANCE_THRESHOLD = 0.55


def clean_llm_output(text: str) -> str:
    """LLM 출력에서 불필요한 마커 제거"""
    for token in ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"]:
        text = text.replace(token, "").strip()
    return text


def get_vectorstore_dimension(vstore_dir: str) -> Optional[int]:
    """Chroma sqlite에서 컬렉션 임베딩 차원 조회"""
    db_path = Path(vstore_dir) / "chroma.sqlite3"
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT dimension FROM collections WHERE name = 'langchain' LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def load_rule_contacts() -> Dict[str, Dict[str, str]]:
    """규정 파일별 담당자 정보 로드"""
    if not CONTACTS_FILE.exists():
        return {}

    try:
        with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def find_article_label(text: str) -> str:
    """청크에서 조항 라벨 추출"""
    match = re.search(r"제\s*\d+\s*조(?:\([^)]+\))?", text)
    if match:
        return match.group(0).replace(" ", "")
    return "관련 조항"


def normalize_korean_text(text: str) -> str:
    """검색/매칭을 위한 단순 정규화 (공백/기호 제거 + 소문자)"""
    lowered = text.lower()
    return re.sub(r"[^0-9a-z가-힣]", "", lowered)


def extract_regulation_name(question: str) -> Optional[str]:
    """질문에서 'OO규정' 형태의 규정명 추출"""
    q = normalize_korean_text(question)
    match = re.search(r"([가-힣a-z0-9]+규정)", q)
    return match.group(1) if match else None


def calc_keyword_overlap(question: str, doc: Document) -> float:
    """질문과 문서 간 키워드 중첩 비율(0~1)"""
    q_tokens = [
        tok for tok in re.findall(r"[가-힣a-zA-Z0-9]{2,}", question.lower())
        if tok not in {"알려줘", "내용", "관련", "규정", "조항", "제조", "제"}
    ]
    if not q_tokens:
        return 0.0

    haystack = (
        normalize_korean_text(Path(doc.metadata.get("source", "")).name)
        + normalize_korean_text(doc.page_content[:1200])
    )
    hit = sum(1 for tok in q_tokens if normalize_korean_text(tok) in haystack)
    return min(1.0, hit / max(len(q_tokens), 1))


def rerank_docs(
    question: str,
    raw_docs: List[Tuple[Document, float]],
) -> List[Tuple[Document, float, float, float, float]]:
    """
    raw_docs: (doc, distance)
    returns: (doc, final_score, semantic_score, lexical_score, distance)
    """
    regulation_name = extract_regulation_name(question)
    regulation_key = normalize_korean_text(regulation_name) if regulation_name else None

    rescored: List[Tuple[Document, float, float, float, float]] = []
    for doc, distance in raw_docs:
        semantic_score = 1.0 / (1.0 + max(distance, 0.0))
        lexical_score = calc_keyword_overlap(question, doc)

        source_name = Path(doc.metadata.get("source", "Unknown")).name
        source_norm = normalize_korean_text(source_name)
        content_norm = normalize_korean_text(doc.page_content[:800])
        source_match = 1.0 if (regulation_key and (regulation_key in source_norm or regulation_key in content_norm)) else 0.0

        final_score = (semantic_score * 0.55) + (lexical_score * 0.25) + (source_match * 0.20)
        rescored.append((doc, final_score, semantic_score, lexical_score, distance))

    rescored.sort(key=lambda x: x[1], reverse=True)

    # 규정명이 질의에 있으면 해당 규정명 매칭 문서를 우선 사용
    if regulation_key:
        matched = []
        for row in rescored:
            doc = row[0]
            source_name = Path(doc.metadata.get("source", "Unknown")).name
            source_norm = normalize_korean_text(source_name)
            content_norm = normalize_korean_text(doc.page_content[:1000])
            if regulation_key in source_norm or regulation_key in content_norm:
                matched.append(row)
        if matched:
            return matched

    return rescored


def format_sources(scored_docs: List[Tuple[Document, float, float, float, float]]) -> str:
    """답변 하단에 붙일 출처 문자열 생성"""
    rows = []
    for idx, (doc, final_score, _, _, _) in enumerate(scored_docs, 1):
        source_name = Path(doc.metadata.get("source", "Unknown")).name
        article = find_article_label(doc.page_content)
        rows.append(f"{idx}. {source_name} - {article} (근거점수: {final_score:.2f})")
    return "\n".join(rows)


def pick_contact_by_sources(
    contacts_map: Dict[str, Dict[str, str]],
    scored_docs: List[Tuple[Document, float, float, float, float]]
) -> Optional[Dict[str, str]]:
    """검색 결과 source 기준으로 담당자 매핑"""
    for doc, *_ in scored_docs:
        source_name = Path(doc.metadata.get("source", "")).name
        if source_name in contacts_map:
            return contacts_map[source_name]

    # source 매칭 실패 시 default 사용
    return contacts_map.get("default")


def format_contact(contact: Optional[Dict[str, str]]) -> str:
    """담당자 문자열 포맷"""
    if not contact:
        return (
            "담당자 정보가 아직 등록되지 않았습니다. "
            "관리자에게 파일별 담당자 정보를 추가해 주세요."
        )

    return (
        f"- 부서: {contact.get('department', '-')}\n"
        f"- 직위: {contact.get('position', '-')}\n"
        f"- 연락처: {contact.get('phone', '-')}\n"
        f"- 이메일: {contact.get('email', '-')}"
    )


@cl.cache
def load_chain():
    """RAG 체인 로드"""

    # 1. 임베딩 로드
    # 기본은 RAGbuilder.get_embeddings() 사용
    # 실패 시 벡터스토어 차원에 맞춰 Ollama 임베딩으로 fallback
    embeddings = None

    try:
        embeddings = get_embeddings(device="CPU")
    except Exception:
        embeddings = None

    if embeddings is None:
        dimension = get_vectorstore_dimension(VSTORE_DIR)

        # 기존 vectorstore 차원에 맞춰 fallback 모델 선택
        # - 2560: qwen3-embedding:4b
        # - 그 외: nomic-embed-text
        fallback_model = "qwen3-embedding:4b" if dimension == 2560 else "nomic-embed-text"
        embeddings = OllamaEmbeddings(model=fallback_model)

    if embeddings is None:
        raise RuntimeError(
            "임베딩 모델을 로드할 수 없습니다. "
            "ollama serve 실행 여부와 임베딩 모델 설치 여부를 확인해주세요."
        )

    # 2. 기존 벡터DB 재사용
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    contacts_map = load_rule_contacts()

    # 3. 문서 요약 정보
    all_docs = db.get()
    sources = {}

    if all_docs and all_docs.get("metadatas"):
        for m in all_docs["metadatas"]:
            source = Path(m.get("source", "Unknown")).name
            sources[source] = sources.get(source, 0) + 1

    doc_summary = {
        "total": len(all_docs["documents"]) if all_docs and all_docs.get("documents") else 0,
        "sources": sources,
    }

    # 4. LLM
    llm = ChatOllama(
        model="qwen2.5:3b-instruct-q4_K_M",
        temperature=0.2,
        num_predict=256,
        num_ctx=1024,
    )

    # 5. 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 한국생산성본부 사규 전용 질의응답 어시스턴트입니다.\n"
            "아래 규칙을 반드시 지키세요.\n"
            "1) 제공된 컨텍스트 내용만 근거로 답변합니다.\n"
            "2) 컨텍스트에 없는 사실, 해석, 일반 상식, 법령 등을 임의로 추가하지 않습니다.\n"
            "3) 질문에 대한 근거가 부족하면 반드시 '규정 근거를 찾지 못했다'고 명시하고 답변을 거절합니다.\n"
            "4) 답변에는 반드시 관련 조항 번호와 핵심 문구를 포함합니다.\n"
            "5) 추측성 표현(예: 아마, 일반적으로, 보통)은 금지합니다.\n\n"
            "컨텍스트:\n{context}\n"
        ),
        ("human", "{question}")
    ])

    def run_qa(question: str) -> str:
        raw_docs = db.similarity_search_with_score(question, k=12)
        rescored_docs = rerank_docs(question, raw_docs)
        filtered_docs = [row for row in rescored_docs if row[1] >= RELEVANCE_THRESHOLD][:4]

        if not filtered_docs:
            contact_text = format_contact(contacts_map.get("default"))
            regulation_name = extract_regulation_name(question)
            regulation_hint = (
                f"\n요청 규정('{regulation_name}')이 벡터DB에 인덱싱되어 있는지도 함께 확인해 주세요."
                if regulation_name else ""
            )
            return (
                "요청하신 질문에 대해 현재 벡터DB에서 신뢰 가능한 사규 근거를 찾지 못했습니다.\n\n"
                "사규에 명시되지 않은 내용은 답변할 수 없습니다. 아래 담당자에게 문의해 주세요.\n"
                f"{regulation_hint}\n"
                f"{contact_text}"
            )

        context_blocks = []
        for idx, (doc, final_score, semantic_score, lexical_score, distance) in enumerate(filtered_docs, 1):
            source_name = Path(doc.metadata.get("source", "Unknown")).name
            context_blocks.append(
                f"[문서{idx}] 파일: {source_name} | 근거점수: {final_score:.2f} "
                f"(semantic={semantic_score:.2f}, lexical={lexical_score:.2f}, distance={distance:.3f})\n"
                f"{doc.page_content}"
            )
        context = "\n\n".join(context_blocks)

        answer_chain = prompt | llm | StrOutputParser() | RunnableLambda(clean_llm_output)
        answer = answer_chain.invoke({"context": context, "question": question})
        sources = format_sources(filtered_docs)

        contact = pick_contact_by_sources(contacts_map, filtered_docs)
        contact_text = format_contact(contact)

        return (
            f"{answer}\n\n"
            f"출처:\n{sources}\n\n"
            "추가 확인이 필요하면 아래 담당자에게 문의해 주세요.\n"
            f"{contact_text}"
        )

    chain = RunnableLambda(run_qa)

    print("KPC 규정 도우미 엔진 로딩 완료")
    return chain, doc_summary


@cl.on_chat_start
async def start():
    """채팅 세션 시작"""
    try:
        chain, doc_summary = load_chain()

        if doc_summary["total"] > 0:
            status_msg = f"현재 {doc_summary['total']}개의 규정 조각이 학습되어 있습니다.\n\n"
            status_msg += "\n".join(
                [f"- {name} ({count} chunks)" for name, count in doc_summary["sources"].items()]
            )
            await cl.Message(
                content=f"🏢 **한국생산성본부 내부규정 도우미**가 준비되었습니다.\n\n{status_msg}"
            ).send()
        else:
            await cl.Message(
                content="⚠️ 학습된 문서가 없습니다. 벡터DB를 먼저 확인해주세요."
            ).send()

        cl.user_session.set("qa_chain", chain)

    except Exception as e:
        await cl.Message(
            content=(
                f"❌ 모델 로딩 중 오류가 발생했습니다: {str(e)}\n\n"
                "확인사항:\n"
                "1. `ollama serve` 실행 여부\n"
                "2. `ollama list` 로 모델 존재 여부 확인\n"
                "3. 벡터DB가 현재 임베딩 모델과 같은 차원으로 생성되었는지 확인"
            )
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """사용자 메시지 처리"""
    chain = cl.user_session.get("qa_chain")

    if not chain:
        await cl.Message(
            content="⚠️ Chain이 초기화되지 않았습니다. 페이지를 새로고침해주세요."
        ).send()
        return

    msg = cl.Message(content="")

    try:
        async for chunk in chain.astream(message.content):
            await msg.stream_token(chunk)
        await msg.send()

    except Exception as e:
        await cl.Message(content=f"❌ 답변 생성 중 오류: {str(e)}").send()
