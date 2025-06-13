from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import sys
from fastapi import FastAPI, Body
from pathlib import Path
import os
from datetime import datetime
# 프로젝트 루트 디렉토리를 Python 경로에 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # app의 상위 디렉토리
sys.path.append(str(PROJECT_ROOT))

from langchain_rag.pre_processing import pre_processing
from langchain_rag.retrieval import VectorStoreRetriever
from langchain_rag.make_prompt import make_prompt
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage


app = FastAPI(title="Tranner RAG API", version="1.0.0")
# VectorStoreRetriever 인스턴스를 애플리케이션 시작 시 한 번만 생성
retriever = VectorStoreRetriever(db_dir=Path("langchain_rag/utils/data/vector_db"))

@app.post("/plan")
def plan_route(
    # 백엔드로부터 오는 원본 JSON 전체를 data로 받습니다.
    data: dict = Body(...)
):
    start_str = data["travel_period"]["start_date"]
    end_str   = data["travel_period"]["end_date"]

    start_dt  = datetime.fromisoformat(start_str)
    end_dt    = datetime.fromisoformat(end_str)

    days = (end_dt - start_dt).days + 1          # 양 끝날 포함
    top_k = days * 7

    print(f"Received data: {data}")
    # 1) 전처리: JSON → 자연어 검색 쿼리 생성
    retrieval_query = pre_processing(data)
    print(f"retrieval_query: {retrieval_query}")
    # 2) 벡터 스토어에서 조회: 상위 10개 장소 정보 반환
    retrieval_output = retriever.retrieve(retrieval_query, top_k)
    print(f"retrieval_output: {retrieval_output}")

    # 3) 프롬프트 생성: RAG 결과(retrieval_output)로 여행 계획 작성용 프롬프트 생성
    prompt = make_prompt(data,retrieval_output)
    print(f"Generated prompt: {prompt}")

    # 4) Claude 호출
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",      # 최신 버전의 Claude 모델
        temperature=0.0,         # 결정론적 응답 옵션
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    # 프롬프트를 HumanMessage로 감싸서 LLM에 전달
    response = llm([HumanMessage(content=prompt)])
    result = response.content
    print(f"LLM response: {result}")
    # 최종 계획을 JSON 응답으로 반환
    return {result}
