# 목표

LangChain과 RAG를 활용하여, 랜덤 여행 계획을 세워주는 프로젝트이다.

# 개요

    백엔드에서 넘겨받는 정보는
        필수:
            1. 인원수
            2. 날짜
            3. 최대 금액
        선택:
            1. 교통수단
            2. 여행 목적 (선호/비선호) - 힐링, 관광, 액티비티, 식사, 핫플, 가족 여행, 커플, 호캉스, 쇼핑, 교육
            3. 숙소 유형 (선호/비선호) - 호텔, 펜션, 에어비앤비, 게스트하우스
            4. 선호 음식 (선호/비선호) - 현지 맛집, 분위기 좋은 카페, 특산물, 고급 레스토랑
    이고,
    넘겨 받은 정보를 활용하여 프롬프트 작성하고, RAG를 활용하여, 랜덤 여행 계획을 세워준다.

# 구성 파일

## store_raw_data.py

    목적: 
        raw data를 store하는데 사용된다.
    방법: 
        raw data에는 구글 api를 사용해서 얻은 값들이다.
        그 값들에는 장소의 이름, 주소, 평점, 가격, 유형, 운영 시간, 리뷰 등이 있다.
        이 값들은 json 형식으로 S3에 저장된다.
        이후 전처리 과정을 통해 한 줄의 문장으로 변환되어 S3에 저장된다.
    후속 처리: 
        store_raw_data.py로 얻은 raw data는 raw_data_split.py로 적절히 분할된다.

## raw_data_split.py

    목적: 
        store_raw_data.py로 얻은 raw data를 적절히 분할한다. (이렇게 문서를 작은 조각으로 나누는 이유는 LLM 모델의 입력 토큰의 개수가 정해져 있기 때문)
    메소드: 
        1. 파일 하나에 저장된 장소 개수를 반환한다.
        2. 파일 하나에 저장된 가장 긴 장소 문장의 단어 수를 반환한다.
        3. 파일 하나에 저장된 모든 장소 목록을 반환한다.
    후속 처리: 
        어떤 장소하나를 가져오기 위해서, 메소드 3번 get_all_places를 사용한다.

embedding.py

3.
	목적:
		split된 data(split data)를 vector로 embedding된다.
	방법:
		OpenAIEmbedding, huggingface embedding 등 여러 가지 방법이 있다.
	후속 처리:
		embedded data를 vector store에 저장한다.

store_vector.py

4.
	목적:
		embedded data(vector)를 vector store에 저장한다.
	방법:
		FAISS, Chroma, milvus(추천) 등의 vector store에 저장할 수 있고, 저장되는 방법 또한 여러 방법이 있을 수 있다.
	후속 처리:
		vector store에 저장된 embedded data를 retriver를 사용하여, 최적의 vector를 검색한다.

retrieval.py

5.
    목적:
        vector store에서 유사한 벡터를 검색하고 유사도를 계산한다.
    방법:
        vector store에서 쿼리와 유사한 벡터를 검색하고, 유사도 점수를 계산한다.
        검색 결과는 유사도에 따라 정렬된다.
    후속 처리:
        검색된 결과는 make_prompt에서 프롬프트 생성에 사용된다.

make_prompt.py

6.
    목적:
        검색된 결과를 바탕으로 LLM에 입력할 프롬프트를 생성한다.
    방법:
        retrieval에서 얻은 검색 결과를 적절한 형식으로 가공하여 프롬프트를 생성한다.
        프롬프트는 시스템 메시지, 사용자 메시지, 컨텍스트 등으로 구성된다.
    후속 처리:
        생성된 프롬프트는 LLM에 입력되어 최종 응답을 생성한다.

post_processing.py

7.
    목적:
        LLM의 출력 결과를 post_processing하여 최종 응답을 생성한다.
    방법:
        LLM의 출력을 정제하여, json 형식으로 변환한다.
    후속 처리:
        후처리된 결과는 최종 사용자에게 전달된다.