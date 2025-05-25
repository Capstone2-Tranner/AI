# app/main.py
from fastapi import FastAPI, Body
from Capstone2.langchain_rag.pre_processing import pre_processing
from Capstone2.langchain_rag.retrieval import VectorStoreRetriever
from Capstone2.langchain_rag.make_prompt import make_prompt
# LangChain Claude 래퍼와 메시지 스키마
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage

app = FastAPI(title="Tranner RAG API", version="1.0.0")

@app.post("/plan")
def plan_route(
    # 백엔드로부터 오는 원본 JSON 전체를 data로 받습니다.
    data: dict = Body(...)
):
    # 1) 전처리 : 받은 JSON을 pre_processing에 넘겨서 retrieval 로 보낼 쿼리 생성
    retrieval_query = pre_processing(data)

    # 2) 벡터 스토어에서 조회
    retrieval_output = retrieve(retrieval_query)

    # 3) 프롬프트 생성 : retrieval_output과 원본 JSON에서 꺼낸 여행 정보로
    prompt = make_prompt(retrieval_output)
    
    # 4) Claude 호출
    llm = ChatAnthropic(
        model="claude-2",      # 또는 "claude-2.0" 등 원하는 모델
        temperature=0.0,       # 결정론적 응답을 원할 땐 0.0
    )
    # HumanMessage 형태로 프롬프트를 감싸서 전달
    response = llm([HumanMessage(content=prompt)])
    result = response.content

    return {"plan": result}

