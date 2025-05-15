'''
1.
    목적: 
        구글 장소 API의 가격 정보를 테스트하는데 사용된다.
    방법: 
        구글 장소 API를 사용하여 특정 위치(서울시청)에서 1개의 place_id를 추출하고,
        그 place_id에 대한 상세 정보를 수집한다.
        즉, Nearby Search API를 1번, Place Details API를 1번 호출한다.
        이 함수 collect_places를 총 num_iterations번 반복하여 장소 정보를 수집한다.
        수집된 정보에는 장소의 이름, 주소, 평점, 가격, 유형, 운영 시간, 리뷰 등이 있다.
        이 정보들은 콘솔에 출력되어 확인된다.
    후속 처리: 
        테스트 결과를 바탕으로 실제 데이터 수집 코드를 작성한다.
'''

from typing import Dict, List
import requests
import os
from dotenv import load_dotenv
import time
# .env 파일에서 환경 변수 로드
load_dotenv()

class PlaceDataCollector:
    def __init__(self):
        self.google_api_key = os.getenv('KHJ_GOOGLE_API_KEY1')
        # 서울시청 좌표로 고정
        self.lat = 37.5665
        self.lng = 126.9780
        self.radius = 1500


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
            "key": self.google_api_key,
            "language": "ko",
            "fields": "name,formatted_address,geometry/location,formatted_phone_number,opening_hours/weekday_text,types,editorial_summary,website,price_level,rating,reviews/rating,reviews/relative_time_description,reviews/text"
        }
        resp = requests.get(url, params=params).json()
        return resp.get("result", {})


    def print_place_details(self, details: Dict):
        """
        장소의 상세 정보를 보기 좋게 출력합니다.
        """
        print("\n=== 장소 상세 정보 ===")
        
        # 기본 정보
        print(f"이름: {details.get('name', 'Unknown')}")
        print(f"주소: {details.get('formatted_address', 'Unknown')}")
        print(f"전화번호: {details.get('formatted_phone_number', 'Unknown')}")
        
        # 위치 정보
        location = details.get('geometry', {}).get('location', {})
        if location:
            print(f"\n위치 정보:")
            print(f"- 위도: {location.get('lat')}")
            print(f"- 경도: {location.get('lng')}")
        
        # 운영 시간
        opening_hours = details.get('opening_hours', {}).get('weekday_text', [])
        if opening_hours:
            print("\n운영 시간:")
            for hours in opening_hours:
                print(f"- {hours}")
        
        # 장소 유형
        types = details.get('types', [])
        if types:
            print(f"\n장소 유형: {', '.join(types)}")
        
        # 편집자 요약
        editorial_summary = details.get('editorial_summary', {})
        if editorial_summary:
            print(f"\n편집자 요약:")
            print(f"- 개요: {editorial_summary.get('overview', 'Unknown')}")
        
        # 웹사이트
        website = details.get('website')
        if website:
            print(f"\n웹사이트: {website}")
        
        # 가격 수준
        price_level = details.get('price_level')
        if price_level is not None:
            price_str = "매우 저렴한" if price_level == 1 else "저렴한" if price_level == 2 else "보통" if price_level == 3 else "비싼" if price_level == 4 else "매우 비싼"
            print(f"\n가격 수준: {price_str}")
        
        # 평점
        rating = details.get('rating')
        if rating:
            print(f"\n평점: {rating}")
        
        # 리뷰
        reviews = details.get('reviews', [])
        if reviews:
            print("\n리뷰:")
            for review in reviews:
                print(f"\n- 평점: {review.get('rating')}")
                print(f"  시간: {review.get('relative_time_description')}")
                print(f"  내용: {review.get('text')}")
        
        print("\n=====================")


    def collect_places(self, num_iterations: int):
        """
        지정된 횟수만큼 반복하여 장소 정보를 수집합니다.
        
        Args:
            num_iterations (int): 반복 횟수
        """
        total_places = 0
        total_details = 0
        
        print(f"\n=== 장소 수집 시작 (총 {num_iterations}회 반복) ===")
        
        for i in range(num_iterations):
            print(f"\n[반복 {i+1}/{num_iterations}]")
            
            # Nearby Search API 호출
            params = {
                "location": f"{self.lat},{self.lng}",
                "radius": self.radius,
                "key": self.google_api_key,
                "type": "cafe"  # 카페 타입으로 검색
            }
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params
            ).json()
            print(resp)
            # 결과에서 첫 번째 place_id만 추출
            if True:
                # place_id = "ChIJv3WOusKifDURkNGSz-MjBAw"
                place_id = resp.get("results", [])[0].get("place_id")
                if place_id:
                    # print(f"Nearby Search 결과: 첫 번째 장소 발견")
                    total_places += 1
                    
                    # 상세 정보 조회
                    details = self.get_google_place_details(place_id)
                    # print(details)
                    if details:
                        total_details += 1
                        print(f"장소 이름 수집: {details.get('name', '이름 없음')}")
                        self.print_place_details(details)
            
            # API 호출 제한을 위한 대기
            # time.sleep(0.1)
        
        print("\n=== 수집 결과 요약 ===")
        # print(f"총 Nearby Search API 호출: {num_iterations}회")
        print(f"총 발견된 장소 수: {total_places}개")
        print(f"총 Place Details API 호출: {total_details}회")
        print("=====================")


def main():
    collector = PlaceDataCollector()
    
    # 테스트할 반복 횟수
    num_iterations = 1  # 원하는 반복 횟수로 조정 가능
    
    # 장소 수집 실행
    collector.collect_places(num_iterations)


if __name__ == "__main__":
    main()

