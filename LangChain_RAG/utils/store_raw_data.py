'''
1.
    목적: 
        raw data를 store하는데 사용된다.
    방법: 
        raw data에는 구글 api를 사용해서 얻은 값들이다.
        그 값들에는 장소의 이름, 주소, 평점, 가격, 유형, 운영 시간, 리뷰 등이 있다.
        이 값들은 json 형식으로 로컬 디스크에 저장된다.
        이후 전처리 과정을 통해 한 줄의 문장으로 변환되어 로컬 디스크에 저장된다.
    후속 처리: 
        store_raw_data.py로 얻은 raw data는 raw_data_split.py로 적절히 분할된다.
'''

import requests
import json
from datetime import datetime
from typing import Dict, List
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # utils의 상위 디렉토리의 상위 디렉토리
sys.path.append(str(PROJECT_ROOT))

from langchain_rag.utils.utils import setup_logger  # utils.py에서 직접 import
from Capstone2.langchain_rag.utils.local_storage import LocalStorage
import math

# .env 파일에서 환경 변수 로드
load_dotenv()

# 현재 폴더 아래에 LOG 폴더 생성
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'LOG')
os.makedirs(log_dir, exist_ok=True)

# 타임스탬프를 포함한 로그 파일명 생성
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'store_raw_data_{timestamp}.log')
logger = setup_logger(__name__, log_file=log_file)

def make_korea_partitions():
    # 제주도 bounding box (Approx.)
    jeju = ((33.288916, 126.162231), (33.458692, 126.942271))

    # 본토 전체 bounding box
    mainland = ((34.354804, 126.128783), (38.614225, 128.356447))
    lat0, lng0 = mainland[0]
    lat1, lng1 = mainland[1]

    # 3x3 그리드로 분할 → 9개 본토 파티션
    lat_steps = 3
    lng_steps = 3
    lat_size = (lat1 - lat0) / lat_steps
    lng_size = (lng1 - lng0) / lng_steps

    parts = []
    for i in range(lat_steps):
        for j in range(lng_steps):
            sw = (lat0 + i * lat_size, lng0 + j * lng_size)
            ne = (lat0 + (i+1) * lat_size, lng0 + (j+1) * lng_size)
            parts.append((sw, ne))

    # 제주도를 마지막 파티션으로 추가 → 총 10개
    parts.append(jeju)
    return parts    # length == 10


def compute_step_deg(sw, ne, grids_per_partition=250):
    lat0, lng0 = sw
    lat1, lng1 = ne
    lat_range = lat1 - lat0
    lng_range = lng1 - lng0

    # 위도와 경도의 비율을 계산
    ratio = lat_range / lng_range
    
    # 위도와 경도의 칸 수를 비율에 맞게 계산
    # lat_steps * lng_steps = grids_per_partition이 되도록
    lat_steps = int(math.sqrt(grids_per_partition * ratio))
    lng_steps = int(grids_per_partition / lat_steps)
    
    # 최소 2칸은 보장
    lat_steps = max(2, lat_steps)
    lng_steps = max(2, lng_steps)
    
    # 각 방향의 step_deg 계산
    step_lat = lat_range / (lat_steps - 1)
    step_lng = lng_range / (lng_steps - 1)
    
    # 더 큰 값을 사용하여 그리드 수를 줄임
    return max(step_lat, step_lng)


def compute_overlap_radius(sw, ne, step_deg, overlap_ratio=0.1):
    """
    격자 간 step_deg에 기반하여 반경을 계산합니다. 대각선 반경이 서로 살짝 겹치도록 설정합니다.
    overlap_ratio: 반경에 추가되는 겹침 비율 (예: 0.2 = 20%)
    """
    lat0, lng0 = sw
    lat1, lng1 = ne
    mid_lat = (lat0 + lat1) / 2.0
    # 위도 1도당 미터 환산
    meter_per_deg_lat = 111320
    # 경도 1도당 미터 환산 (중간 위도 기준)
    meter_per_deg_lng = 40075000 * math.cos(math.radians(mid_lat)) / 360
    step_lat_m = step_deg * meter_per_deg_lat
    step_lng_m = step_deg * meter_per_deg_lng
    half_diagonal = math.sqrt(step_lat_m**2 + step_lng_m**2) / 2
    return int(half_diagonal * (1 + overlap_ratio))


class PlaceDataCollector:
    """
    장소 정보를 수집하는 클래스
    구글 api를 통해 장소 정보를 수집하고 통합
    """

    def __init__(self, worker_id: int):
        # worker_id에 따른 Google API 키 설정
        self.google_api_key = os.getenv(f'GOOGLE_API_KEY_{worker_id}')
        if not self.google_api_key:
            raise ValueError(f"GOOGLE_API_KEY_{worker_id} not found in environment variables")
            
        # 전역 중복 방지를 위한 place_id 집합
        self.seen_ids = set()
        # 전체 수집된 장소 수 추적
        self.total_places = 0
        # LocalStorage 인스턴스 생성
        self.storage = LocalStorage()


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

    # Nearby Search API를 1번 사용, 최대 20개의 place_id를 details API로 조회
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
        각 격자점 결과는 로컬 디스크에 JSON과 전처리된 텍스트로 저장됩니다.
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
        
        print(f"max_grids: {max_grids}")
    
        # 격자점 카운터
        grid_counter = 0
        # 전체 수집된 장소 수 초기화
        self.total_places = 0

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
                
                # 한 그리드당 단일 Nearby Search 호출: 최대 20개 place_id 처리
                new_count = 0
                logger.info(f"위도: {lat:.3f}, 경도: {lng:.3f} (격자점: {grid_counter + 1}/{max_grids})")

                # API 요청 파라미터 설정 (한 번만 호출)
                params = {
                    "location": f"{lat},{lng}",
                    "radius": radius,
                    "key": self.google_api_key
                }
                resp = requests.get(
                    "https://maps.googleapis.com/maps/api/place/nearbysearch/json", # 5000회 무료
                    params=params
                ).json()

                # 최대 20개 결과만 사용
                page_results = resp.get("results", [])[:20]
                page_pids = []
                for item in page_results:
                    pid = item.get("place_id")
                    if pid and pid not in self.seen_ids:
                        self.seen_ids.add(pid)
                        page_pids.append(pid)
                if page_pids:
                    details_list = self.get_multiple_place_details_parallel(page_pids)
                    for detail in details_list:
                        if "대한민국" in detail.get("formatted_address", ""):
                            current_grid_places.append(detail)
                            new_count += 1
                            self.total_places += 1

                logger.info(f"\t이 격자점에서 수집된 신규 장소 개수: {new_count}개")
                
                # 현재 격자점의 모든 데이터를 파일로 저장
                if current_grid_places:
                    self.save_grid_data(current_grid_places, grid_counter+1, current_grid_info)
                
                # 다음 경도로 이동
                lng += step_deg
                grid_counter += 1
            # 다음 위도로 이동
            lat += step_deg
        
        logger.info(f"파티션 처리 완료: 총 {self.total_places}개의 장소 수집됨")


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


    def save_grid_data(self, places: List[Dict], grid_number: int, grid_info: Dict):
        """
        한 격자점의 모든 데이터를 로컬 디스크에 저장합니다.
        
        Args:
            places (List[Dict]): 저장할 장소 데이터 리스트
            grid_number (int): 격자점 번호
            grid_info (Dict): 격자점 정보 (위도, 경도, 반경 등)
        """
        # JSON 데이터 저장
        grid_data = {"grid_info": grid_info, "places": places}
        grid_json = json.dumps(grid_data, ensure_ascii=False, indent=4)
        # 그리드 번호 대신 (위도,경도) 조합으로 고유 인덱스 사용
        lat_str = f"{grid_info['lat']:.3f}"
        lng_str = f"{grid_info['lng']:.3f}"
        raw_path = f"raw_data/raw_data_{lat_str}_{lng_str}.json"
        self.storage.save_to_disk(grid_json.encode('utf-8'), raw_path)
        logger.info(f"\t격자점 ({grid_info['lat']:.3f},{grid_info['lng']:.3f}) 원본 데이터 저장 완료: {raw_path}")


def main():
    # 그리드당 Nearby Search 호출 1회, detail API 호출 20회 => 총 20개의 장소 정보 수집
    # grids_per_partition = 500  # 500개의 격자점 처리 => Nearby Search 호출 500회, detail API 호출 10,000회 => Nearby Search 무료, detail API 90 달러
    # grids_per_partition = 1000  # 1000개의 격자점 처리 => Nearby Search 호출 1,000회, detail API 호출 20,000회 => Nearby Search 무료, detail API 180 달러
    grids_per_partition = 10  # 1500개의 격자점 처리 => Nearby Search 호출 1,500회, detail API 호출 30,000회 => Nearby Search 무료, detail API 270 달러
    # grids_per_partition = 2000  # 2000개의 격자점 처리 => Nearby Search 호출 2,000회, detail API 호출 40,000회 => Nearby Search 무료, detail API 360 달러

    try:
        # 파티션별로 step_deg와 radius를 동적 계산하여 호출
        partitions = make_korea_partitions()

        # 1부터 10까지 순차적으로 처리
        for worker_id in range(10, 11):  # 1~10
            print(f"\n=== Processing Worker {worker_id} ===")
            
            # 각 worker_id에 맞는 collector 인스턴스 생성
            collector = PlaceDataCollector(worker_id)
            
            partition_idx = worker_id - 1  # worker_id에 해당하는 파티션 인덱스

            # 해당 작업자에게 할당된 파티션만 처리
            sw, ne = partitions[partition_idx]
            step_deg_val = compute_step_deg(sw, ne, grids_per_partition)
            radius_val = compute_overlap_radius(sw, ne, step_deg_val)
            print(f"worker_id: {worker_id}, partition_idx: {partition_idx}")
            print(f"sw: {sw}, ne: {ne}, step_deg_val: {step_deg_val}, radius_val: {radius_val}")
            collector.get_almost_korean_places(
                start_lat=sw[0], end_lat=ne[0],
                start_lng=sw[1], end_lng=ne[1],
                step_deg=step_deg_val, radius=radius_val
            )
            print(f"=== Completed Worker {worker_id} ===\n")
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}")


if __name__ == "__main__":
    main()