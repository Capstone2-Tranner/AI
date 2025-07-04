'''
2.
    목적: 
        store_raw_data.py로 얻은 raw data를 적절히 분할한다. (이렇게 문서를 작은 조각으로 나누는 이유는 LLM 모델의 입력 토큰의 개수가 정해져 있기 때문)
    메소드: 
        1. 파일 하나에 저장된 장소 개수를 반환한다.
        2. 파일 하나에 저장된 가장 긴 장소 문장의 단어 수를 반환한다.
        3. 파일 하나에 저장된 모든 장소 목록을 반환한다.
    후속 처리: 
        어떤 장소 하나를 가져오기 위해서 embdding하기 위해, 메소드 3번 get_all_places를 사용한다.
'''

import os
from typing import List, Dict
from pathlib import Path
import json
import numpy as np
import sys
import shutil

# 프로젝트 루트 디렉토리를 Python 경로에 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# store_vector.py에서 필요한 함수들 import
from store_vector import create_hnsw_index, save_metadata

class LocalStorage:
    def __init__(self, base_dir: str = None):
        """
        로컬 파일 시스템 데이터 처리를 위한 클래스 초기화
        
        Args:
            base_dir (str): 기본 디렉토리 경로 (기본값: 현재 디렉토리)
        """
        self.base_dir = Path(base_dir) if base_dir else CURRENT_DIR
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

    # 목적: 특정 폴더에서 지정된 개수만큼의 파일 목록을 가져옵니다.
    # 예시: list_first_n_files("data/", 5) -> ["data/file1.txt", "data/file2.txt", "data/file3.txt", "data/file4.txt", "data/file5.txt"]
    def list_first_n_files(self, folder_path: str, n: int) -> List[str]:
        """
        특정 폴더 내에 있는 파일 경로 중 앞에서부터 n개를 반환합니다.

        Args:
            folder_path (str): 폴더 경로 (예: 'preprocessed_raw_data/')
            n (int): 반환할 파일 개수

        Returns:
            List[str]: 파일 경로 목록 (최대 n개)
        """
        folder_path = self.base_dir / folder_path
        if not folder_path.exists():
            return []

        files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                files.append(str(file_path.relative_to(self.base_dir)))
                if len(files) >= n:
                    break
        return files

    # 목적: 특정 폴더 내의 모든 파일 목록을 가져옵니다.
    # 예시: list_files_in_folder("data/") -> ["data/file1.txt", "data/file2.txt", "data/subfolder/file3.txt"]
    def list_files_in_folder(self, folder_path: str) -> List[str]:
        """
        특정 폴더 내에 있는 모든 파일의 경로를 반환합니다.
        
        Args:
            folder_path (str): 폴더 경로 (예: 'preprocessed_raw_data/')
        
        Returns:
            List[str]: 파일 경로 목록
        """
        folder_path = self.base_dir / folder_path
        if not folder_path.exists():
            return []

        files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                files.append(str(file_path.relative_to(self.base_dir)))
        return files

    # 목적: 파일의 내용을 문자열로 읽어옵니다.
    # 예시: get_file_content("data/example.txt") -> "파일의 내용..."
    def get_file_content(self, file_path: str) -> str:
        """
        파일 내용을 읽어옵니다.
        
        Args:
            file_path (str): 파일 경로
        
        Returns:
            str: 파일 내용
        """
        file_path = self.base_dir / file_path
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        return file_path.read_text(encoding='utf-8')

    # 목적: 텍스트 파일에서 장소 정보 블록의 개수를 세어 반환합니다.
    # 예시: count_places("data/places.txt") -> 42
    def count_places(self, file_path: str) -> int:
        """
        전처리된 데이터 파일에서 장소 문장 개수를 계산하여 반환합니다.
        
        Args:
            file_path (str): 전처리된 데이터 파일 경로
        
        Returns:
            int: 장소 문장 개수
        """
        content = self.get_file_content(file_path)
        blocks = [bl for bl in content.split('\n\n') if bl.strip()]
        place_blocks = blocks[1:]
        return len(place_blocks)

    # 목적: 텍스트 파일에서 가장 긴 장소 설명의 단어 수를 찾습니다.
    # 예시: longest_place_word_length("data/places.txt") -> 156
    def longest_place_word_length(self, file_path: str) -> int:
        """
        전처리된 데이터 파일에서 가장 긴 장소 문장의 단어 수를 반환합니다.
        
        Args:
            file_path (str): 전처리된 데이터 파일 경로
        
        Returns:
            int: 가장 긴 장소 문장의 단어 수
        """
        content = self.get_file_content(file_path)
        blocks = [bl for bl in content.split('\n\n') if bl.strip()]
        place_blocks = blocks[1:]
        max_len = 0
        for place in place_blocks:
            word_count = len(place.split())
            if word_count > max_len:
                max_len = word_count
        return max_len

    # 목적: 텍스트 파일에서 모든 장소 정보를 리스트로 추출합니다.
    # 예시: get_all_places_from_txt("data/places.txt") -> ["장소1 정보...", "장소2 정보...", ...]
    def get_all_places_from_txt(self, file_path: str) -> List[str]:
        """
        전처리된 데이터 파일에서 모든 장소 문장을 리스트로 반환합니다.
        
        Args:
            file_path (str): 전처리된 데이터 파일 경로
        
        Returns:
            List[str]: 장소 문장 리스트
        """
        content = self.get_file_content(file_path)
        blocks = [bl for bl in content.split('\n\n') if bl.strip()]
        place_blocks = blocks[1:]
        return place_blocks

    # 목적: JSON 파일에서 장소 정보를 구조화된 형식으로 추출합니다.
    # 예시: get_all_places_from_json("data/places.json") -> ["장소명: 카페\n주소: 서울시...", ...]
    def get_all_places_from_json(self, file_path: str) -> List[str]:
        """
        JSON 형식의 원본 데이터 파일에서 모든 장소 정보를 추출합니다.
        
        Args:
            file_path (str): JSON 파일 경로
            
        Returns:
            List[str]: 장소 정보 리스트
        """
        content = self.get_file_content(file_path)
        data = json.loads(content)
        places = []
        
        if 'places' in data:
            for place in data['places']:
                place_info = []
                if 'name' in place:
                    place_info.append(f"장소명: {place['name']}")
                if 'formatted_address' in place:
                    place_info.append(f"주소: {place['formatted_address']}")
                if 'geometry' in place and 'location' in place['geometry']:
                    location = place['geometry']['location']
                    place_info.append(f"위치: 위도 {location['lat']}, 경도 {location['lng']}")
                if 'types' in place:
                    place_info.append(f"유형: {', '.join(place['types'])}")
                if 'rating' in place:
                    place_info.append(f"평점: {place['rating']}")
                if 'reviews' in place:
                    reviews = place['reviews']
                    if reviews:
                        place_info.append("리뷰:")
                        for review in reviews[:2]:
                            if 'text' in review:
                                place_info.append(f"- {review['text'][:100]}...")
                
                if place_info:
                    places.append('\n'.join(place_info))
        
        return places

    def get_all_places_from_predata(self, file_path: str) -> List[str]:
        """
        전처리된 텍스트 형식의 데이터 파일에서 모든 장소 정보를 추출합니다.

        Args:
            file_path (str): 텍스트 파일 경로

        Returns:
            List[str]: 장소 정보 리스트
        """
        content = self.get_file_content(file_path)
        places = []

        # 빈 줄(또는 여러 개의 공백 문자)로 구분된 엔트리 단위로 분리
        entries = [block.strip() for block in content.split("\n\n") if block.strip()]

        for entry in entries:
            place_info = []
            # 한 엔트리 안의 각 라인을 “키 : 값” 형태로 분리
            for line in entry.splitlines():
                # 콜론 기준으로 앞뒤로 나누되, 키와 값 모두 strip() 처리
                if ":" not in line:
                    continue
                key, value = map(str.strip, line.split(":", 1))

                if key == "주소":
                    place_info.append(f"주소: {value}")
                elif key == "위도, 경도":
                    # "위도, 경도   : 33.349888, 126.2560608" -> 위도, 경도 분할
                    lat_lng = [coord.strip() for coord in value.split(",")]
                    if len(lat_lng) == 2:
                        place_info.append(f"위치: 위도 {lat_lng[0]}, 경도 {lat_lng[1]}")
                elif key == "이름":
                    place_info.append(f"장소명: {value}")
                elif key == "전체 평점":
                    place_info.append(f"평점: {value}")
                elif key == "타입":
                    # "supermarket, grocery_or_supermarket, ..." 형태
                    types = [t.strip() for t in value.split(",")]
                    place_info.append(f"유형: {', '.join(types)}")
                elif key == "한 줄 요약 리뷰":
                    # 리뷰가 "리뷰 없음"인 경우에도 그대로 추가하거나,
                    # 필요에 따라 예외 처리를 해줄 수 있음
                    summary = value if value != "리뷰 없음" else "리뷰 없음"
                    place_info.append(f"리뷰 요약: {summary}")

            if place_info:
                places.append("\n".join(place_info))

        return places


    # 목적: 바이너리 데이터를 파일로 저장합니다.
    # 예시: save_to_disk(b"Hello World", "data/example.bin")
    def save_to_disk(self, data: bytes, file_path: str) -> None:
        """
        데이터를 파일로 저장하는 메소드
        
        Args:
            data (bytes): 저장할 데이터
            file_path (str): 저장할 파일 경로
        """
        file_path = self.base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        print(f"[SAVE] 파일 저장 완료: {file_path}")

    # 목적: 텍스트와 벡터를 파일로 저장하여 검색 인덱스를 생성합니다.
    # 예시: save_vector_store(["텍스트1", "텍스트2"], vectors_array, "embedding/")
    def save_vector_store(self, texts: List[str], vectors: np.ndarray, base_path: str = "embedding") -> None:
        """
        벡터와 메타데이터를 파일로 저장하는 메소드
        
        Args:
            texts (List[str]): 저장할 텍스트 목록
            vectors (np.ndarray): 저장할 벡터
            base_path (str): 저장할 기본 경로
        """
        base_path = self.base_dir / base_path
        base_path.mkdir(parents=True, exist_ok=True)
        
        # HNSW 인덱스 저장
        hnsw_path = base_path / "hnsw.index"
        create_hnsw_index(vectors, str(hnsw_path))
        
        # 메타데이터 저장
        meta_path = base_path / "metadata.json"
        meta_bytes = json.dumps(texts).encode('utf-8')
        self.save_to_disk(meta_bytes, str(meta_path.relative_to(self.base_dir)))

def main():
    """
    모든 함수를 테스트하는 메인 함수
    """
    # LocalStorage 인스턴스 생성
    storage = LocalStorage()

    # 테스트용 디렉토리 및 파일 생성
    test_dir = "test_data"
    test_txt = f"{test_dir}/test_places.txt"
    test_json = f"{test_dir}/test_places.json"

    # 테스트용 데이터 생성
    test_txt_content = """메타데이터
장소1: 서울시 강남구의 카페
주소: 서울시 강남구 테헤란로 123
평점: 4.5

장소2: 부산시 해운대구의 레스토랑
주소: 부산시 해운대구 해운대해변로 456
평점: 4.8"""

    test_json_content = {
        "places": [
            {
                "name": "테스트 카페",
                "formatted_address": "서울시 강남구 테헤란로 123",
                "geometry": {"location": {"lat": 37.123, "lng": 127.456}},
                "types": ["cafe", "food"],
                "rating": 4.5,
                "reviews": [
                    {"text": "좋은 카페입니다. 분위기가 좋아요."},
                    {"text": "커피가 맛있어요."}
                ]
            }
        ]
    }

    # 테스트 데이터 저장
    storage.save_to_disk(test_txt_content.encode('utf-8'), test_txt)
    storage.save_to_disk(json.dumps(test_json_content, ensure_ascii=False).encode('utf-8'), test_json)

    print("\n" + "="*50)
    print("1. 파일 목록 테스트")
    print("="*50)
    # list_first_n_files 테스트
    print("\n[list_first_n_files 테스트]")
    first_n_files = storage.list_first_n_files(test_dir, 2)
    print(f"처음 2개 파일: {first_n_files}")

    # list_files_in_folder 테스트
    print("\n[list_files_in_folder 테스트]")
    all_files = storage.list_files_in_folder(test_dir)
    print(f"모든 파일: {all_files}")

    print("\n" + "="*50)
    print("2. 텍스트 파일 처리 테스트")
    print("="*50)
    # count_places 테스트
    print("\n[count_places 테스트]")
    place_count = storage.count_places(test_txt)
    print(f"장소 개수: {place_count}")

    # longest_place_word_length 테스트
    print("\n[longest_place_word_length 테스트]")
    max_words = storage.longest_place_word_length(test_txt)
    print(f"가장 긴 장소 설명 단어 수: {max_words}")

    # get_all_places_from_txt 테스트
    print("\n[get_all_places_from_txt 테스트]")
    txt_places = storage.get_all_places_from_txt(test_txt)
    print(f"텍스트 파일 장소 목록 (총 {len(txt_places)}개):")
    for i, place in enumerate(txt_places, 1):
        print(f"{i}. {place}")

    print("\n" + "="*50)
    print("3. JSON 파일 처리 테스트")
    print("="*50)
    # get_all_places_from_json 테스트
    print("\n[get_all_places_from_json 테스트]")
    json_places = storage.get_all_places_from_json(test_json)
    print(f"JSON 파일 장소 목록 (총 {len(json_places)}개):")
    for i, place in enumerate(json_places, 1):
        print(f"{i}. {place}")

    print("\n" + "="*50)
    print("4. 벡터 저장소 테스트")
    print("="*50)
    # save_vector_store 테스트
    print("\n[save_vector_store 테스트]")
    test_texts = ["테스트 텍스트 1", "테스트 텍스트 2"]
    test_vectors = np.random.rand(2, 128)  # 2개의 128차원 벡터
    storage.save_vector_store(test_texts, test_vectors, f"{test_dir}/test_vectors")

    # 테스트 데이터 정리
    print("\n" + "="*50)
    print("테스트 완료 및 정리")
    print("="*50)
    try:
        shutil.rmtree(test_dir)
        print("테스트 데이터 정리 완료")
    except Exception as e:
        print(f"테스트 데이터 정리 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 