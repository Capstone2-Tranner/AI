==================== 목표 ====================

LangChain과 RAG를 활용하여, 랜덤 여행 계획을 세워주는 프로젝트이다.

==================== 개요 ====================

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

==================== 구성 파일 ====================

store_raw_data_2_s3_by_multiple_thread.py

1. 목적:
    Google Places API로 수집한 raw 데이터를 멀티스레드로 빠르게 S3에 저장합니다.
2. 방법:
    - 지정한 위·경도 범위(start_lat~end_lat, start_lng~end_lng)를 격자(step_deg) 단위로 순회합니다.
    - 각 격자점에서 radius(m)를 반경으로 nearbysearch를 호출해 place_id를 수집합니다.
    - 수집된 place_id를 ThreadPoolExecutor로 병렬 상세 조회 후 대한민국 필터링합니다.
    - (위도,경도) 조합을 파일명으로 JSON과 전처리된 텍스트를 S3에 업로드합니다.
3. 주요 파라미터:
    - start_lat, end_lat: 검색할 위도 범위 (기본 33.0~38.5)
    - start_lng, end_lng: 검색할 경도 범위 (기본 125.0~130.0)
    - step_deg: 격자 간격(도, 기본 0.018)
    - radius: 검색 반경(미터, 기본 1500)
    - max_grids: 최대 격자점 개수(None=전체 범위)
4. 실행 예시:
    # 전체 범위 기본 호출
    collector.get_almost_korean_places()

    # 3명 분할 실행
    MSJ) collector.get_almost_korean_places(start_lat=33.0, end_lat=34.8, start_lng=125.0, end_lng=130.0)
    KMS) collector.get_almost_korean_places(start_lat=34.8, end_lat=36.6, start_lng=125.0, end_lng=130.0)
    KMW) collector.get_almost_korean_places(start_lat=36.6, end_lat=38.5, start_lng=125.0, end_lng=130.0)

    # 5명 분할 실행
    MSJ) collector.get_almost_korean_places(start_lat=33.0, end_lat=34.1, start_lng=125.0, end_lng=130.0)
    KMS) collector.get_almost_korean_places(start_lat=34.1, end_lat=35.2, start_lng=125.0, end_lng=130.0)
    KMW) collector.get_almost_korean_places(start_lat=35.2, end_lat=36.3, start_lng=125.0, end_lng=130.0)
    CEB) collector.get_almost_korean_places(start_lat=36.3, end_lat=37.4, start_lng=125.0, end_lng=130.0)
    KHJ) collector.get_almost_korean_places(start_lat=37.4, end_lat=38.5, start_lng=125.0, end_lng=130.0)

raw_data_split.py

2.
    목적: 
        raw_data_load로 얻은 raw data를 적절히 분할한다.
    방법: 
        구역 단위(구단위 or 시단위)로 장소 데이터를 split 하거나, 300줄로 split 하거나 여러 경우의 수가 있다.
        이 부분에서 여러 경우로 테스트 해봐서 최적의 split 방법을 골라야한다.
    후속 처리:
       split된 data(split data)를 embedding.py를 통해 vector로 embedding한다.

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