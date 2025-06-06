def pre_processing(data: dict) -> str:
    """
    주어진 여행 설문조사 JSON 데이터에서
    - 주소 (region)
    - 선호 유형 (travel_style_preferences.prefer)
    - 비선호 유형 (travel_style_preferences.nonPrefer)
    형식으로 추출하여 최종 retrieval 쿼리 문자열을 생성합니다.
    """
    # 1. 주소 정보 추출
    region = data.get("region", "").strip()
    address_line = f"주소 : {region}" if region else "주소 : 정보 없음"

    # 2. 여행 스타일 정보 추출
    styles = data.get("travel_style_preferences", {})
    prefers = styles.get("prefer", [])
    non_prefers = styles.get("nonPrefer", [])

    # 선호 유형이 없을 때
    if prefers:
        prefer_str = ", ".join(prefers)
    else:
        prefer_str = "선호 유형 없음"

    # 비선호 유형이 없을 때
    if non_prefers:
        non_prefer_str = ", ".join(non_prefers)
    else:
        non_prefer_str = "비선호 유형 없음"

    prefer_line = f"선호 타입 : {prefer_str}"
    non_prefer_line = f"비선호 타입 : {non_prefer_str}"

    # 3. 최종 쿼리 문자열 합치기
    query = "\n".join([address_line, prefer_line, non_prefer_line])
    return query


if __name__ == "__main__":
    # 테스트용 예시 데이터
    sample_data = {
        "travel_period": {
            "start_date": "2025-07-10",
            "end_date":   "2025-07-15"
        },
        "group": {
            "num_people": 4,
            "ages": ["adult", "adult", "child", "elder"]
        },
        "budget": {
            "min": 500000,
            "max": 1000000
        },
        "region": "제주 서귀포시",
        "transportation_preferences": [
            "public_transport",
            "vehicle"
        ],
        "travel_style_preferences": {
            "prefer": [
                "힐링",
                "액티비티",
                "맛집 탐방"
            ],
            "nonPrefer": [
                "쇼핑"
            ]
        }
    }

    result = pre_processing(sample_data)
    print(result)
