from fastapi import FastAPI, Body
from pathlib import Path

from Capstone2.langchain_rag.pre_processing import pre_processing
from Capstone2.langchain_rag.retrieval import VectorStoreRetriever
from Capstone2.langchain_rag.make_prompt import make_prompt
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage

app = FastAPI(title="Tranner RAG API", version="1.0.0")
# VectorStoreRetriever 인스턴스를 애플리케이션 시작 시 한 번만 생성
retriever = VectorStoreRetriever(db_dir=Path("data/vector_db"))

@app.post("/plan")
def plan_route(
    # 백엔드로부터 오는 원본 JSON 전체를 data로 받습니다.
    data: dict = Body(...)
):
    # 1) 전처리: JSON → 자연어 검색 쿼리 생성
    retrieval_query = pre_processing(data)

    # 2) 벡터 스토어에서 조회: 상위 10개 장소 정보 반환
    retrieval_output = retriever.retrieve(retrieval_query, top_k=10)

    # 3) 프롬프트 생성: RAG 결과(retrieval_output)로 여행 계획 작성용 프롬프트 생성
    prompt = make_prompt(retrieval_output)

    # 4) Claude 호출
    llm = ChatAnthropic(
        model="claude-2",      # 또는 "claude-2.0" 등
        temperature=0.0,         # 결정론적 응답 옵션
    )
    # 프롬프트를 HumanMessage로 감싸서 LLM에 전달
    response = llm([HumanMessage(content=prompt)])
    result = response.content

    # 최종 계획을 JSON 응답으로 반환
    return {"plan": result}
