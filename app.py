import chainlit as cl
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

# 개선된 RAGbuilder 사용 (기존 Qwen3-Embedding 재활용)
from RAGbuilder_improved import get_embeddings, VSTORE_DIR

def clean_llm_output(text: str) -> str:
    """Qwen2.5 출력에서 불필요한 마커 제거"""
    # 혹시나 하는 정리
    for token in ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"]:
        text = text.replace(token, "").strip()
    return text

@cl.cache
def load_chain():
    """Qwen2.5 + 기존 Qwen3-Embedding 조합 (추가 다운로드 없음)"""
    
    # 1. 임베딩 - 기존 Qwen3-Embedding 재활용 (추가 다운로드 X)
    embeddings = get_embeddings(device="CPU")
    if embeddings is None:
        raise RuntimeError("임베딩 모델을 로드할 수 없습니다. RAGbuilder_improved.py validate 명령으로 환경을 확인해주세요.")
    
    # 2. 기존 벡터DB 재사용
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    # 3. 문서 요약 정보
    all_docs = db.get()
    sources = {}
    if all_docs and all_docs['metadatas']:
        for m in all_docs['metadatas']:
            source = Path(m.get('source', 'Unknown')).name
            sources[source] = sources.get(source, 0) + 1
    doc_summary = {
        "total": len(all_docs['documents']) if all_docs else 0,
        "sources": sources
    }
    
    # 4. LLM - ollama의 Qwen2.5 (한국어 특화)
    llm = ChatOllama(
        model="qwen2.5:3b-instruct-q4_K_M",
        temperature=0.2,
        num_predict=256,
        num_ctx=1024,
    )
    
    # 5. 프롬프트 - 기존과 동일
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "# 역할 및 목표 "
         "당신은 한국생산성본부의 사규, 규정, 규칙, 가이드라인 등 내부 지침을 숙지한 AI 어시스턴트입니다. 당신의 주요 임무는 "
         "1. 사용자가 특정 상황과 함께 해결하고 싶은 문제나 고민을 제시하면 이에 적용될 수 있는 규정의 이름과 조항 번호 등을 제시합니다. "
         "2. 사용자가 특정 상황과 규정을 제시하며 특정 상황을 해당 규정으로 해결할 수 있는지 질문하면, 문헌적 해석을 우선하여 답변하되, 문헌적 해석이 불가능한 경우에 한하여 대한민국 관련 법령 및 합리적 추론을 활용하여 답변합니다. "
         "3. 모든 답변은 {context}를 토대로 제시하고, 답변에 참고가 된 내용이 무엇인지 근거가 되는 조항(예: 인사규정 제1조제2항, 복무규정 제3조제4항제5호 등)을 반드시 밝혀주세요. "
         "# 주의사항 "
         "답변에 근거가 없거나 그 결과물이 확실하지 않다면, 없는 내용을 지어내는 것보다 모른다고 답하는 것이 더 낫습니다. "
         "근거가 되는 규정들이 상충되거나 해석의 여지가 있다면, 그 점을 밝히고 담당부서에 문의하라고 안내해주세요."),
        ("human", "{question}")
    ])
    
    # 6. Chain 구성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_llm_output)
    )
    
    print("🏢 KPC 규정 도우미 엔진 로딩 완료! (Qwen2.5 + Qwen3-Embedding)")
    return chain, doc_summary

# --- Chainlit 이벤트 핸들러 ---

@cl.on_chat_start
async def start():
    """채팅 세션 시작"""
    try:
        chain, doc_summary = load_chain()
        
        if doc_summary["total"] > 0:
            status_msg = f"현재 {doc_summary['total']}개의 규정 조각이 학습되어 있습니다.\n\n"
            status_msg += "\n".join([f"- {name} ({count} chunks)" for name, count in doc_summary["sources"].items()])
            await cl.Message(content=f"🏢 **한국생산성본부 내부규정 도우미**가 준비되었습니다.\n\n{status_msg}").send()
        else:
            await cl.Message(content="⚠️ 학습된 문서가 없습니다. 벡터DB를 먼저 확인해주세요.").send()
        
        cl.user_session.set("qa_chain", chain)
    
    except Exception as e:
        await cl.Message(content=f"❌ 모델 로딩 중 오류가 발생했습니다: {str(e)}\n\n**ollama 서버가 실행 중인지 확인해주세요**: `ollama serve`").send()

@cl.on_message
async def main(message: cl.Message):
    """사용자 메시지 처리"""
    chain = cl.user_session.get("qa_chain")
    
    if not chain:
        await cl.Message(content="⚠️ Chain이 초기화되지 않았습니다. 페이지를 새로고침해주세요.").send()
        return
    
    msg = cl.Message(content="")
    
    try:
        async for chunk in chain.astream(message.content, config={}):
            await msg.stream_token(chunk)
        await msg.send()
    
    except Exception as e:
        await cl.Message(content=f"❌ 답변 생성 중 오류: {str(e)}").send()
