'''
1.
    목적: 
        raw data를 load하는데 사용된다.
    방법: 
        raw data에는 네이버 장소 api를 사용해서 얻은 자료일 수 있고, pdf일 수도 있고, 여러 경우의 수가 있다.
    후속 처리: 
        raw_data_load로 얻은 raw data는 raw_data_split을 통해 적절히 분할된다.
'''

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import time

# .env 파일에서 환경 변수 로드
load_dotenv()

class PlaceDataCollector:
    """
    장소 정보를 수집하는 클래스
    다양한 API(네이버, 카카오, 구글, T맵)를 통해 장소 정보를 수집하고 통합
    """

    def __init__(self):
        # 환경 변수에서 각 API의 키값을 가져옴
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        self.kakao_api_key = os.getenv('KAKAO_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.tmap_api_key = os.getenv('TMAP_API_KEY')


    def get_naver_place_info(self, query: str) -> Dict:
        """
        네이버 지도 API를 통해 장소 정보 수집
        Args:
            query (str): 검색할 장소 이름
        Returns:
            Dict: 네이버 API에서 반환한 JSON 데이터
        """
        url = "https://openapi.naver.com/v1/search/local.json"
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret
        }
        params = {
            "query": query,
            "display": 5  # 최대 5개의 결과만 가져옴
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()


    def get_kakao_place_info(self, query: str) -> Dict:
        """
        카카오맵 API를 통해 장소 정보 수집
        Args:
            query (str): 검색할 장소 이름
        Returns:
            Dict: 카카오 API에서 반환한 JSON 데이터
        """
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {
            "Authorization": f"KakaoAK {self.kakao_api_key}"
        }
        params = {
            "query": query
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    

    def get_google_place_info(self, query: str, num_results: int = 5) -> List[str]:
        """
        Google Text Search API를 통해 장소 검색
        Args:
            query (str): 검색할 장소 이름
            num_results (int): 가져올 결과의 최대 개수
        Returns:
            List[str]: 검색된 장소들의 place_id 리스트
        """
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query":    query,
            "key":      self.google_api_key,
            "language": "ko"  # 한국어로 결과 반환
        }
        data = requests.get(url, params=params).json()
        # 최대 num_results개의 place_id만 추출
        return [item["place_id"] for item in data.get("results", [])[:num_results]]


    def get_google_place_details(self, place_id: str) -> Dict:
        """
        Google Place Details API를 통해 장소의 상세 정보 수집
        Args:
            place_id (str): 구글 플레이스 ID
        Returns:
            Dict: 장소의 상세 정보
        """
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "key":      self.google_api_key,
            "language": "ko",
            # 필요한 필드만 지정하여 API 응답 최적화
            # "fields":   "name,formatted_address,geometry,opening_hours,reviews,website"
            "fields":   "name,formatted_address,geometry/location,opening_hours/weekday_text,website,reviews"
        }
        resp = requests.get(url, params=params).json()
        return resp.get("result", {})


    def get_tmap_route_info(self, start_lat: float, start_lon: float, 
                           end_lat: float, end_lon: float) -> Dict:
        """
        T맵 API를 통해 두 지점 사이의 경로 정보 수집
        Args:
            start_lat (float): 출발지 위도
            start_lon (float): 출발지 경도
            end_lat (float): 도착지 위도
            end_lon (float): 도착지 경도
        Returns:
            Dict: 경로 정보
        """
        url = "https://apis.openapi.sk.com/tmap/routes"
        headers = {
            "appKey": self.tmap_api_key
        }
        params = {
            "startX": start_lon,
            "startY": start_lat,
            "endX": end_lon,
            "endY": end_lat,
            "reqCoordType": "WGS84GEO"  # WGS84 좌표계 사용
        }
        response = requests.post(url, headers=headers, data=params)
        return response.json()


    def combine_place_data(self, place_name: str) -> Dict:
        """
        여러 API의 데이터를 통합하여 하나의 장소 정보로 구성
        Args:
            place_name (str): 검색할 장소 이름
        Returns:
            Dict: 통합된 장소 정보
        """
        # Google API를 통해 장소 검색 및 상세 정보 수집
        place_ids = self.get_google_place_info(place_name, num_results=5)
        print("==================== place_ids ====================")
        print(place_ids)
        google_details = [self.get_google_place_details(pid) for pid in place_ids]
        print("==================== google_details ====================")
        print(google_details)
        
        # 수집된 정보를 통합하여 하나의 데이터 구조로 구성
        combined_data = {
            "name":            place_name,
            # "naver_info":      naver_data,
            # "kakao_info":      kakao_data,
            "google_info":     google_details,
            "operating_hours": self._extract_operating_hours(google_details),
            "location":        self._extract_location(google_details),
            "reviews":         self._combine_reviews(google_details),
            "last_updated":    datetime.now().isoformat()  # 데이터 수집 시간 기록
        }
        return combined_data


    def _extract_operating_hours(self, google_details: List[Dict]) -> Dict:
        """
        운영 시간 정보 추출
        Args:
            google_details (List[Dict]): 구글 API에서 가져온 장소 상세 정보 리스트
        Returns:
            Dict: 추출된 운영 시간 정보
        """
        # 운영 시간 정보를 각 API에서 추출하여 통합
        return {
            # "naver": naver_data.get("operating_hours", {}),
            # "kakao": kakao_data.get("operating_hours", {}),
            "google": [g.get("opening_hours", {}) for g in google_details]
        }


    def _extract_location(self, google_details: List[Dict]) -> Dict:
        """
        위치 정보 추출
        Args:
            google_details (List[Dict]): 구글 API에서 가져온 장소 상세 정보 리스트
        Returns:
            Dict: 추출된 위치 정보
        """
        return {
            # "address": {
            #     "naver": naver_data.get("address", ""),
            #     "kakao": kakao_data.get("address", "")
            # },
            # "coordinates": {
            #     "latitude": kakao_data.get("y", ""),
            #     "longitude": kakao_data.get("x", "")
            # },
            "google": [
                g.get("geometry", {}).get("location", {}) 
                for g in google_details
            ]
        }


    def _combine_reviews(self, google_details: List[Dict]) -> List[Dict]:
        """
        리뷰 정보 통합
        Args:
            google_details (List[Dict]): 구글 API에서 가져온 장소 상세 정보 리스트
        Returns:
            List[Dict]: 통합된 리뷰 정보 리스트
        """
        reviews = []
        # if "reviews" in naver_data:
        #     reviews.extend(naver_data["reviews"])
        # if "reviews" in kakao_data:
        #     reviews.extend(kakao_data["reviews"])
        for g in google_details:
            reviews.extend(g.get("reviews", []))
        return reviews


    def get_all_korean_places(self, step_deg: float = 0.1, radius: int = 15000, max_grids: int = ((38.5-33.0)/0.1+1) * ((130.0-125.0)/0.1+1)) -> List[Dict]:
        """
        한국 전역을 격자 기반으로 검색하여 모든 장소 상세 정보를 수집합니다.
        
        Args:
            step_deg (float): 격자 간격(도 단위). 기본값 0.5도
            radius (int): 각 격자점에서 검색할 반경(미터). 기본값 50km
            max_grids (int): 검색할 최대 격자점 개수. 기본값 10개
            
        Returns:
            List[Dict]: 수집된 장소 상세 정보 리스트
            
        동작 방식:
            1. 한국의 영역을 위도 33-38.5도, 경도 125-130도로 설정
            2. 설정된 격자 간격(step_deg)으로 순회하며 각 지점에서 검색 수행
            3. 각 격자점에서 지정된 반경(radius) 내의 장소들을 검색
            4. 페이지네이션을 처리하여 모든 결과를 수집
            5. 중복 방지를 위해 이미 수집된 place_id는 건너뜀
            6. max_grids 개수만큼만 격자점 검색
        """
        """
        | 반경(radius) | step_deg (≈) | 격자 한 변 길이 (도→km) |
        |---------|-----------------------------|---------------------------|
        | 1 000 m | ≒ 2 000 / 111 320 = 0.018° |  0.018° × 111.32 ≒   2 km |
        |   500 m | ≒ 1 000 / 111 320 = 0.009° |  0.009° × 111.32 ≒   1 km |
        |   250 m | ≒  500 / 111 320 = 0.0045° | 0.0045° × 111.32 ≒ 0.5 km |
        |   100 m | ≒  200 / 111 320 = 0.0018° | 0.0018° × 111.32 ≒ 0.2 km |
        |    50 m | ≒  100 / 111 320 = 0.0009° | 0.0009° × 111.32 ≒ 0.1 km |
        """
        # 한국의 지리적 범위 설정 (위도/경도)
        min_lat, max_lat = 33.0, 38.5  # 위도: 제주도 남단 ~ 휴전선 북단
        min_lng, max_lng = 125.0, 130.0  # 경도: 서해안 ~ 동해안

        # 중복 방지를 위한 이미 수집된 장소 ID 저장 집합
        seen_ids = set()
        # 수집된 모든 장소 정보를 저장할 리스트
        all_details = []
        # 검색한 격자점 개수를 세는 카운터
        grid_count = 0

        # 위도 방향 순회
        lat = min_lat
        while lat <= max_lat and grid_count < max_grids:
            # 경도 방향 순회
            lng = min_lng
            while lng <= max_lng and grid_count < max_grids:
                # 페이지네이션 토큰 초기화
                next_page_token = None
                new_count = 0
                # 격자점 정보 출력 (한 번)
                print(f"위도: {lat:.1f}, 경도: {lng:.1f} (격자점: {grid_count + 1}/{max_grids})")
                while True:
                    # API 요청 파라미터 설정
                    params = {
                        "location": f"{lat},{lng}",  # 현재 격자점의 위치
                        "radius": radius,  # 검색 반경
                        "key": self.google_api_key  # API 키
                    }
                    if next_page_token:
                        params["pagetoken"] = next_page_token
                        
                    # Google Places API 호출
                    resp = requests.get(
                        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                        params=params
                    ).json()
                    
                    # 검색 결과 처리
                    for item in resp.get("results", []):
                        pid = item.get("place_id")
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            # 장소 상세 정보 조회 후 국가 필터 (대한민국)
                            detail = self.get_google_place_details(pid)
                            # formatted_address에서 대한민국이 포함되어 있는지 확인
                            if "대한민국" in detail.get("formatted_address", ""):
                                all_details.append(detail)
                                new_count += 1
                            
                    # 다음 페이지 토큰 확인
                    next_page_token = resp.get("next_page_token")
                    if not next_page_token:
                        break
                    # API 제한 준수를 위한 대기
                    # time.sleep(2)
                # 이 격자점에서 수집된 신규 장소 개수 출력
                print(f"이 격자점에서 수집된 신규 장소 개수: {new_count}개")
                # 다음 경도로 이동
                lng += step_deg
                grid_count += 1
            # 다음 위도로 이동
            lat += step_deg

        return all_details


def save_raw_data(data: Dict, filename: str):
    """
    수집된 데이터를 JSON 파일로 저장합니다.
    저장 폴더는 스크립트 파일 기준의 raw_data 디렉토리입니다.
    """
    # 스크립트 파일 위치 기준의 raw_data 폴더 경로 생성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)
    # 저장할 파일 경로
    filepath = os.path.join(raw_dir, f"{filename}.json")
    # 파일 저장 및 경로 출력
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"데이터 저장 완료: {filepath}")


def main():
    """
    메인 실행 함수
    여러 장소에 대한 데이터를 수집하고 저장
    """
    collector = PlaceDataCollector()
    
    try:
        place_data = collector.get_all_korean_places(max_grids=25)
        save_raw_data(place_data, "korean_places")
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()