'''
7.
    목적:
        LLM의 출력 결과를 post_processing하여 최종 응답을 생성한다.
    방법:
        LLM의 출력을 정제하여, json 형식으로 변환한다.
    후속 처리:
        후처리된 결과는 최종 사용자에게 전달된다.
'''
def pre_processing(data: dict) -> str:
    """
    Convert survey JSON data into a natural language query string for retrieval.
    """
    # "2025-07-10" 같은 ISO 형식의 날짜 문자열을 -로 분리해 (2025, 7, 10) 형태의 정수 튜플로 반환하는 역할할
    def format_date(date_str: str) -> tuple:
        year, month, day = date_str.split('-')
        return int(year), int(month), int(day)

    # 날짜 정보 추출
    start_y, start_m, start_d = format_date(data["travel_period"]["start_date"]) # 출발 날짜 연/월/일 반환
    end_y, end_m, end_d = format_date(data["travel_period"]["end_date"]) # 도착 날짜 연/월/일 반환

    start_str = f"{start_y}년 {start_m}월 {start_d}일" # 출발 날짜 문자열 생성
    if end_y == start_y:
        end_str = f"{end_m}월 {end_d}일" # 출발 날짜와 같은 연도면 연도 생략함함
    else:
        end_str = f"{end_y}년 {end_m}월 {end_d}일"

    date_segment = f"{start_str}부터 {end_str}까지" # 출발일부터 도착일까지의 날짜 문자열 생성

    # 인원 정보 추출출
    ages = data.get("group", {}).get("ages", [])
    total_people = len(ages) # 총 인원 수 계산
    age_map = {"adult": "성인", "child": "어린이", "elder": "노약자"}
    counts = {key: ages.count(key) for key in age_map} # ages 리스트에서 "adult", "child", "elder" 개수 세어 "성인 2명, 어린이 1명" 와 같은 형태로 만든다다
    age_details = ", ".join(
        f"{age_map[key]} {count}명" for key, count in counts.items() if count > 0
    )
    group_segment = f"총 {total_people}명({age_details})" # 전체 인원수(성인 2명, 어린이 1명) 같은 형태의 문자열 생성성

    # 지역 정보 추출
    region = data.get("region", "").strip() # 지역 이름 추출해서 ~~로 여행을 계획합니다 형태로 만든다다
    region_segment = f"{region}로 여행을 계획합니다"

    # 예산 정보 추출
    min_budget = data.get("budget", {}).get("min", 0) # 최소 예산
    max_budget = data.get("budget", {}).get("max", 0) # 최대 예산
    def format_budget(value: int) -> str:
        if value % 10000 == 0: # 가격이 만 원 단위로 딱 떨어지면
            return f"{value // 10000}만 원" # 50만원 같은 형태로 반환
        return f"{value:,}원" # 가격이 만 원 단위로 딱 떨어지지 않으면 52,300원 같은 형태로 반환

    budget_segment = (
        f"예산은 최소 {format_budget(min_budget)}에서 최대 {format_budget(max_budget)}이며"
    )

    # 교통 수단 정보 추출
    transport_map = {"public_transport": "대중교통", "vehicle": "차량"}
    # 설문 조사를 교통 수단 리스트에서 각 항목을 매핑하여 "대중교통, 차량" 같은 형태로 만든다
    transports = [transport_map.get(t, t) for t in data.get("transportation_preferences", [])]
    if len(transports) > 1:
        transport_str = ", ".join(transports[:-1]) + "과 " + transports[-1]
    else:
        transport_str = transports[0] if transports else ""
    transport_segment = f"{transport_str}을 선호합니다"

    # 여행 스타일 정보 추출
    styles = data.get("travel_style_preferences", {})
    prefers = styles.get("prefer", [])
    non_prefers = styles.get("nonPrefer", [])
    prefer_str = ", ".join(prefers)
    non_prefer_str = ", ".join(non_prefers)
    style_segment = (
        f"{prefer_str}을 선호하고 {non_prefer_str}은 비선호합니다"
    )

    # 최종 쿼리문 생성성
    query = (
        f"{date_segment} {group_segment}이 {region_segment}. "
        f"{budget_segment} {transport_segment}. {style_segment}."
    )
    return query
