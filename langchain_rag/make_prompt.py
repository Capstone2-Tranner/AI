from typing import List, Dict

def make_prompt(request_data: Dict, retrieval_output: List[str]) -> str:
    """
    Claude에게 전달할 여행 일정 계획 프롬프트를 생성합니다.

    Args:
        request_data: 프론트엔드로부터 받은 원본 요청 데이터. 형식 예시:
            {
                "travel_period": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
                "group": {"num_people": int, "ages": [str, ...]},
                "budget": {"min": int, "max": int},
                "region": str,
                "transportation_preferences": [str, ...],
                "travel_style_preferences": {"prefer": [str], "nonPrefer": [str]}
            }
        retrieval_output: VectorStoreRetriever로부터 반환된 장소 리스트. 각 요소는
            "주소: ...\n위치: ...\n장소명: ...\n평점: ...\n유형: ...\n리뷰 요약: ..."

    Returns:
        Claude에게 보낼 프롬프트 문자열 (한국어)
    """
    # 1. 요청 데이터에서 여행 정보 추출
    period = request_data.get("travel_period", {})
    start_date = period.get("start_date", "")
    end_date = period.get("end_date", "")
    num_people = request_data.get("group", {}).get("num_people", "")
    region = request_data.get("region", "")
    transport = request_data.get("transportation_preferences", [])
    style_pref = request_data.get("travel_style_preferences", {})
    prefer = style_pref.get("prefer", [])
    non_prefer = style_pref.get("nonPrefer", [])

    # 2. 기본 프롬프트
    prompt = []
    prompt.append("당신은 여행 일정 계획 전문가 AI입니다.")
    prompt.append(f"여행 인원: {num_people}명")
    prompt.append(f"여행 기간: {start_date}부터 {end_date}까지")
    prompt.append(f"여행 지역: {region}")
    if transport:
        prompt.append(f"교통 수단 선호: {', '.join(transport)}")
    if prefer or non_prefer:
        prompt.append(
            f"여행 스타일 선호: {', '.join(prefer)}",)
        prompt.append(
            f"비선호 스타일: {', '.join(non_prefer)}")
    prompt.append("\n### 검색된 장소 정보(상위 10개) ###")

    # 3. retrieval_output 나열
    for idx, info in enumerate(retrieval_output, start=1):
        prompt.append(f"{idx}. {info}")
    # 4. Hallucination 방지 지침 추가
    prompt.append("\n### 주의사항 ###")
    prompt.append("- 검색된 장소 정보 이외의 어떠한 정보도 추가하지 마세요.")
    prompt.append("- 제공된 데이터 외에는 절대 새로 생성하지 마세요.")
    prompt.append("- 필드에 값이 없으면 '알 수 없음'으로 표시하세요.")

    # 5. JSON 출력 요구사항
    prompt.append("위 검색 결과를 바탕으로, 여행 일정을 다음 JSON 구조에 맞춰 작성하세요.")
    prompt.append("하루에 장소는 최소 3개 이상 포함되어야 하며, 각 장소는 식당, 관광지, 숙소 등 다양한 유형을 포함해야 합니다.")
    prompt.append("- 응답은 오직 JSON으로만 이루어져야 합니다. 추가 설명 금지.")
    prompt.append("- 필드 이름과 구조를 정확히 지켜야 합니다.")
    prompt.append("- 시간은 HH:MM 형식, 위도/경도는 숫자, 순번은 정수로 표시하세요.")
    prompt.append("### JSON 예시 ###")
    prompt.append(
"{\n  \"country_name\": \"대한민국\",\n  \"region_name\": \"서울\",\n  \"start_date\": \"" + start_date + "\",\n  \"end_date\": \"" + end_date + "\",\n  \"detailSchedule\": [\n    {\n      \"day_seq\": 1,\n      \"scheduleByDay\": [\n        {\n          \"location_seq\": 1,\n          \"start_time\": \"08:00\",\n          \"end_time\": \"09:30\",\n                  \"place_name\": \"경복궁\",\n          \"place_type\": \"tourist_attraction\",\n          \"address\": \"서울특별시 종로구 사직로 161\",\n          \"latitude\": 37.579617,\n          \"longitude\": 126.977041\n        }\n      ]\n    }\n  ]\n}")

    # 결합 후 반환
    return "\n".join(prompt)
