'''
1.
    목적: 
        raw data를 store하는데 사용된다.
    방법: 
        raw data에는 구글 api를 사용해서 얻은 값들이다. 이때, multi-thread를 사용하여 빠르게 수집한다.
        그 값들에는 장소의 이름, 주소, 평점, 가격, 유형, 운영 시간, 리뷰 등이 있다.
        이 값들은 json 형식으로 저장된다.
        이후 전처리 과정을 통해 한 줄의 문장으로 변환되어 S3에 저장된다.
    후속 처리: 
        store_raw_data_2_s3_by_multiple_thread.py로 얻은 raw data는 raw_data_split.py로 적절히 분할된다.
'''

import requests
import json
from datetime import datetime
from typing import Dict, List
import os
from dotenv import load_dotenv
import boto3
from concurrent.futures import ThreadPoolExecutor
from utils import setup_logger
import time

# .env 파일에서 환경 변수 로드
load_dotenv()
logger = setup_logger(__name__, log_file="store_raw_data_2_s3_by_multiple_thread.log")

class PlaceDataCollector:
    """
    장소 정보를 수집하는 클래스
    구글 api를 통해 장소 정보를 수집하고 통합
    """

    def __init__(self):
        # 환경 변수에서 API 키 및 S3 버킷 이름을 가져옴
        self.google_api_key = os.getenv('KMS_GOOGLE_API_KEY')
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        # AWS 자격증명 및 리전 설정
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'ap-northeast-2')
        # S3 클라이언트 생성
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )


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
            "fields":   "name,formatted_address,geometry/location,opening_hours/weekday_text,types,website,editorial_summary,price_level,rating,reviews/rating,reviews/relative_time_description,reviews/text"
        }
        resp = requests.get(url, params=params).json()
        return resp.get("result", {})


    def get_almost_korean_places(
        self,
        start_lat: float = 33.0,
        end_lat: float = 38.5,
        start_lng: float = 125.0,
        end_lng: float = 130.0,
        step_deg: float = 0.018,
        radius: int = 1500,
        max_grids: int = None
    ) -> None:
        """
        지정한 범위(start_lat~end_lat, start_lng~end_lng)를 격자 기반으로 검색하여 장소 정보를 수집합니다.
        각 격자점 결과는 S3에 JSON과 전처리된 텍스트로 저장됩니다.
        Args:
            start_lat (float): 검색 시작 위도
            end_lat (float): 검색 종료 위도
            start_lng (float): 검색 시작 경도
            end_lng (float): 검색 종료 경도
            step_deg (float): 격자 간격(도)
            radius (int): 검색 반경(미터)
            max_grids (int, optional): 최대 격자점 개수, None일 경우 전체 범위
        """
        # 검색 범위 설정 (start/end lat, lng)
        min_lat, max_lat = start_lat, end_lat
        min_lng, max_lng = start_lng, end_lng
        # max_grids가 None이면 주어진 범위로부터 계산된 격자점 개수 사용
        if max_grids is None:
            # 세로(lat) 방향 격자 수 계산
            lat_steps = int((end_lat - start_lat) / step_deg) + 1
            # 가로(lng) 방향 격자 수 계산
            lng_steps = int((end_lng - start_lng) / step_deg) + 1
            max_grids = lat_steps * lng_steps

        # 중복 방지를 위한 이미 수집된 장소 ID 저장 집합
        seen_ids = set()
        # 격자점 카운터
        grid_counter = 0

        # 위도 방향 순회
        lat = min_lat
        while lat <= max_lat and grid_counter < max_grids:
            # 경도 방향 순회
            lng = min_lng
            while lng <= max_lng and grid_counter < max_grids:
                # 현재 격자점의 장소들을 저장할 리스트
                current_grid_places = []
                # 현재 격자점 정보 저장
                current_grid_info = {
                    "lat": lat,
                    "lng": lng,
                    "radius": radius,
                    "step_deg": step_deg,
                    "grid_number": grid_counter
                }
                
                # 페이지네이션 토큰 초기화 및 place_id 수집 리스트
                next_page_token = None
                new_count = 0
                all_pids = []
                logger.info(f"위도: {lat:.3f}, 경도: {lng:.3f} (격자점: {grid_counter + 1}/{max_grids})")

                while True:
                    
                    # API 요청 파라미터 설정
                    params = {
                        "location": f"{lat},{lng}",
                        "radius": radius,
                        "key": self.google_api_key
                    }
                    if next_page_token:
                        params["pagetoken"] = next_page_token

                    # Google Places API 호출
                    resp = requests.get(
                        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                        params=params
                    ).json()

                    # place_id 수집 및 병렬 상세 조회
                    page_pids = []
                    for item in resp.get("results", []):
                        pid = item.get("place_id")
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            all_pids.append(pid)
                            page_pids.append(pid)
                    if page_pids:
                        details_list = self.get_multiple_place_details_parallel(page_pids)
                        for detail in details_list:
                            if "대한민국" in detail.get("formatted_address", ""):
                                current_grid_places.append(detail)
                                new_count += 1

                    # 다음 페이지 토큰 확인
                    next_page_token = resp.get("next_page_token")
                    if not next_page_token:
                        break
                    # 페이지네이션 토큰 활성화를 위해 잠시 대기
                    time.sleep(1)
                
                logger.info(f"\t이 격자점에서 수집된 신규 장소 개수: {new_count}개")
                
                # 현재 격자점의 모든 데이터를 파일로 저장
                if current_grid_places:
                    self.save_grid_data_2_s3(current_grid_places, grid_counter+1, current_grid_info)
                
                # 다음 경도로 이동
                lng += step_deg
                grid_counter += 1
            # 다음 위도로 이동
            lat += step_deg


    def get_multiple_place_details_parallel(self, place_ids: List[str], max_workers: int = 5) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pid = {
                executor.submit(self.get_google_place_details, pid): pid
                for pid in place_ids
            }
            results = []
            for future in future_to_pid:
                try:
                    details = future.result()
                    results.append(details)
                except Exception as e:
                    pid = future_to_pid[future]
                    logger.error(f"Error fetching details for {pid}: {e}")
            return results


    def save_grid_data_2_s3(self, places: List[Dict], grid_number: int, grid_info: Dict):
        """
        한 격자점의 모든 데이터를 S3 버킷에 업로드합니다.
        
        Args:
            places (List[Dict]): 저장할 장소 데이터 리스트
            grid_number (int): 격자점 번호
            grid_info (Dict): 격자점 정보 (위도, 경도, 반경 등)
        """
        # S3에 JSON 데이터 업로드
        grid_data = {"grid_info": grid_info, "places": places}
        grid_json = json.dumps(grid_data, ensure_ascii=False, indent=4)
        # 그리드 번호 대신 (위도,경도) 조합으로 고유 인덱스 사용
        lat_str = f"{grid_info['lat']:.3f}"
        lng_str = f"{grid_info['lng']:.3f}"
        raw_key = f"raw_data/raw_data_{lat_str}_{lng_str}.json"
        self.s3_client.put_object(Bucket=self.s3_bucket_name, Key=raw_key, Body=grid_json.encode('utf-8'), ContentType='application/json')
        logger.info(f"\tS3에 격자점 ({grid_info['lat']:.3f},{grid_info['lng']:.3f}) 데이터 업로드 완료: s3://{self.s3_bucket_name}/{raw_key}")

        # 전처리된 문장 생성 및 S3 업로드
        sentences = preprocess_raw_data_to_sentences(places)
        sentence_lines = [f"격자점 정보: 위도 {grid_info['lat']:.3f}, 경도 {grid_info['lng']:.3f}, 반경 {grid_info['radius']}m", ""]
        sentence_lines += sentences
        sentence_content = "\n\n".join(sentence_lines)
        # 전처리 파일명도 동일한 방식으로 생성
        pre_key = f"preprocessed_raw_data/preprocessed_data_{lat_str}_{lng_str}.txt"
        self.s3_client.put_object(Bucket=self.s3_bucket_name, Key=pre_key, Body=sentence_content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
        logger.info(f"\tS3에 전처리된 격자점 ({grid_info['lat']:.3f},{grid_info['lng']:.3f}) 데이터 업로드 완료: s3://{self.s3_bucket_name}/{pre_key}")



def preprocess_raw_data_to_sentences(place_data: List[Dict]) -> List[str]:
    """
    장소 데이터를 한 줄의 문장 형식으로 전처리합니다.
    각 장소의 모든 정보가 하나의 문장으로 통합됩니다.
    
    Args:
        place_data (List[Dict]): 수집된 장소 데이터 리스트
        
    Returns:
        List[str]: 한 줄로 변환된 장소 정보 리스트
    """
    
    sentences = []
    
    for place in place_data:
        # 기본 정보 추출
        name = place.get("name", "")
        address = place.get("formatted_address", "")
        rating = place.get("rating", 0)
        price_level = place.get("price_level", 0)
        types = place.get("types", [])
        
        # 운영 시간 정보 추출
        opening_hours = place.get("opening_hours", {}).get("weekday_text", [])
        
        # 리뷰 정보 추출 및 통합
        reviews = place.get("reviews", [])
        review_texts = []
        for review in reviews[:5]:  # 최대 5개의 리뷰만 포함
            review_text = review.get("text", "").replace("\n", " ").strip()
            review_rating = review.get("rating", 0)
            if review_text:
                review_texts.append(f"'{review_text}' (평점: {review_rating})")
        
        # 문장 구성
        sentence_parts = []
        
        # 기본 정보 추가
        if name and address:
            sentence_parts.append(f"{name}은(는) {address}에 위치한")
        
        # 장소 유형 추가
        if types:
            type_str = ", ".join(types)
            sentence_parts.append(f"{type_str}입니다")
        
        # 평점 정보 추가
        if rating:
            sentence_parts.append(f"평점은 {rating}점입니다.")
        
        # 가격 수준 정보 추가
        if price_level:
            price_str = "매우 저렴한" if price_level == 1 else "저렴한" if price_level == 2 else "보통" if price_level == 3 else "비싼" if price_level == 4 else "매우 비싼"
            sentence_parts.append(f"가격 수준은 {price_str} 편입니다.")
        
        # 운영 시간 정보 추가
        if opening_hours:
            hours_str = ", ".join(opening_hours)
            sentence_parts.append(f"운영 시간은 {hours_str}입니다.")
        
        # 리뷰 정보 추가
        if review_texts:
            reviews_str = ", ".join(review_texts)
            sentence_parts.append(f"리뷰는 다음과 같습니다: {reviews_str}")
        
        # 모든 부분을 하나의 문장으로 통합
        sentence = " ".join(sentence_parts) + "."
        sentences.append(sentence)
    
    return sentences



def main():
    """
    메인 실행 함수
    여러 장소에 대한 데이터를 수집하고 저장

    실행 예시 (3명 분할):
        MSJ) collector.get_almost_korean_places(start_lat=33.0, end_lat=34.8, start_lng=125.0, end_lng=130.0) 
        KMS) collector.get_almost_korean_places(start_lat=34.8, end_lat=36.6, start_lng=125.0, end_lng=130.0) 
        KMW) collector.get_almost_korean_places(start_lat=36.6, end_lat=38.5, start_lng=125.0, end_lng=130.0) 

    실행 예시 (5명 분할):
        MSJ) collector.get_almost_korean_places(start_lat=33.0, end_lat=34.1, start_lng=125.0, end_lng=130.0) 
        KMS) collector.get_almost_korean_places(start_lat=34.1, end_lat=35.2, start_lng=125.0, end_lng=130.0) 
        KMW) collector.get_almost_korean_places(start_lat=35.2, end_lat=36.3, start_lng=125.0, end_lng=130.0) 
        CEB) collector.get_almost_korean_places(start_lat=36.3, end_lat=37.4, start_lng=125.0, end_lng=130.0) 
        KHJ) collector.get_almost_korean_places(start_lat=37.4, end_lat=38.5, start_lng=125.0, end_lng=130.0) 
    """

    collector = PlaceDataCollector()
    
    try:
        # 데이터 수집 및 그리드 단위로 저장
        # collector.get_almost_korean_places(start_lat=33.0, end_lat=38.5, start_lng=125.0, end_lng=130.0)
        # collector.get_almost_korean_places()
        collector.get_almost_korean_places(step_deg=0.1, radius=15000, max_grids=25)
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}")



if __name__ == "__main__":
    main()