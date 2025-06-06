'''
6.
    목적:
        검색된 결과를 바탕으로 LLM에 입력할 프롬프트를 생성한다.
    방법:
        retrieval에서 얻은 검색 결과를 적절한 형식으로 가공하여 프롬프트를 생성한다.
        프롬프트는 시스템 메시지, 사용자 메시지, 컨텍스트 등으로 구성된다.
    후속 처리:
        생성된 프롬프트는 LLM에 입력되어 최종 응답을 생성한다.
'''
# utils/make_prompt.py

def make_prompt(data: dict) -> str:
    """
    여행 일정 생성을 위한 프롬프트 문자열을 생성합니다.
    - data: {
        "places": [ {name, address, location, type, rating, reviewSummary}, ... ],
        "howManyPeople": int or str,
        "startDate": "YYYY-MM-DD",
        "endDate": "YYYY-MM-DD"
      }
    반환값: Claude에게 전달할 프롬프트 문자열 (한국어)
    """
    # 1. 딕셔너리에서 값 추출
    places = data.get("places", [])
    people = data.get("howManyPeople", "")
    start = data.get("startDate", "")
    end   = data.get("endDate", "")

    # 2. 기본 여행 정보 소개
    prompt = (
        f"당신은 여행 일정 계획 전문가 AI입니다.\n"
        f"여행 인원: {people}명\n"
        f"여행 기간: {start}부터 {end}까지\n\n"
    )

    # 3. 추천 장소 상세 정보 나열
    prompt += "### 추천 여행지 정보\n"
    for idx, place in enumerate(places, start=1):
        prompt += (
            f"{idx}. 장소명: {place.get('name','')}\n"
            f"   주소: {place.get('address','')}\n"
            f"   위치: {place.get('location','')}\n"
            f"   유형: {place.get('type','')}\n"
            f"   평점: {place.get('rating','')}\n"
            f"   리뷰 요약: {place.get('reviewSummary','')}\n\n"
        )
    # 4. 출력 형식 및 요구 사항 명시 (수정된 부분)
    prompt += (
        "### 요구 사항\n"
        "- 위 정보를 바탕으로 여행 일정을 계획하세요.\n"
        "- **JSON 형식**으로만 결과를 출력해야 합니다 (추가 설명 금지).\n"
        "- JSON 구조는 다음 예시를 따르세요:\n"
        "{\n"
        '  "country_name": "대한민국",\n'
        '  "region_name": "서울",\n'
        f'  "start_date": "{start}",\n'
        f'  "end_date": "{end}",\n'
        '  "detailSchedule": [\n'
        '    {\n'
        '      "day_seq": 1,\n'
        '      "scheduleByDay": [\n'
        '        {\n'
        '          "location_seq": 1,\n'
        '          "start_time": "08:00",\n'
        '          "end_time": "09:30",\n'
        '          "place_id": "ChIJ...abc",\n'
        '          "place_name": "경복궁",\n'
        '          "place_type": "tourist_attraction",\n'
        '          "address": "서울특별시 종로구 사직로 161",\n'
        '          "latitude": 37.579617,\n'
        '          "longitude": 126.977041\n'
        '        },\n'
        '        {\n'
        '          "location_seq": 2,\n'
        '          "start_time": "10:00",\n'
        '          "end_time": "10:30",\n'
        '          "place_id": "ChIJ...xyz",\n'
        '          "place_name": "카페온",\n'
        '          "place_type": "cafe",\n'
        '          "address": "서울특별시 마포구 양화로 45",\n'
        '          "latitude": 37.556345,\n'
        '          "longitude": 126.922071\n'
        '        }\n'
        '      ]\n'
        '    },\n'
        '    {\n'
        '      "day_seq": 2,\n'
        '      "scheduleByDay": [\n'
        '        {\n'
        '          "location_seq": 1,\n'
        '          "start_time": "08:00",\n'
        '          "end_time": "09:30",\n'
        '          "place_id": "ChIJ...pqr",\n'
        '          "place_name": "N서울타워",\n'
        '          "place_type": "tourist_attraction",\n'
        '          "address": "서울특별시 용산구 남산공원길 105",\n'
        '          "latitude": 37.551169,\n'
        '          "longitude": 126.988226\n'
        '        },\n'
        '        {\n'
        '          "location_seq": 2,\n'
        '          "start_time": "10:00",\n'
        '          "end_time": "10:30",\n'
        '          "place_id": "ChIJ...stu",\n'
        '          "place_name": "스타벅스 명동점",\n'
        '          "place_type": "cafe",\n'
        '          "address": "서울특별시 중구 명동길 26",\n'
        '          "latitude": 37.563005,\n'
        '          "longitude": 126.982679\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "- 위 예시와 같은 필드 이름(`country_name`, `region_name`, `start_date`, `end_date`, `detailSchedule` 등)과\n"
        "  구조를 반드시 지켜주세요.\n"
        "- `day_seq`와 `location_seq`는 정수로, `latitude`/`longitude`는 숫자로, 시간(`start_time`/`end_time`)은 HH:MM 형식으로 출력합니다.\n"
        "- 응답은 오직 JSON 데이터만 출력하세요. 불필요한 문장이나 코드블럭을 포함하지 마세요.\n"
    )

    return prompt
