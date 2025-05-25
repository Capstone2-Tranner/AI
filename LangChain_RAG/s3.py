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

# 상위 디렉토리의 모듈 import를 위한 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
STORE_VECTOR_PATH = CURRENT_DIR.parent / "store_vector" / "store_vector.py"

# store_vector.py에서 필요한 함수들 import
sys.path.append(str(STORE_VECTOR_PATH.parent))
from langchain_rag.store_vector import create_hnsw_index, save_metadata

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
    분석 실행 함수
    """
    # LocalStorage 인스턴스 생성
    storage = LocalStorage()

    folder_path = "embedding/"
    file_path = "embedding/metadata.json"

    files = storage.list_files_in_folder(folder_path)
    print("\n" + "="*50)
    print("폴더 내 파일 목록")
    print("="*50)
    for file in files:
        print(f"• {file}")

    print("\n" + "="*50)
    print(f"파일 분석: {file_path}")
    print("="*50)
    
    # 파일 내용 출력
    content = storage.get_file_content(file_path)
    if file_path.endswith('.json'):
        print(json.dumps(json.loads(content), indent=2, ensure_ascii=False))
        places = storage.get_all_places_from_json(file_path)
    else:
        print(content)
        places = storage.get_all_places_from_txt(file_path)

    print("\n" + "="*50)
    print(f"장소 목록 (총 {len(places)}개)")
    print("="*50)
    for i, place in enumerate(places, 1):
        print(f"{i:02d}. {place}\n")
        
    print("="*50)
    print("분석 결과")
    print("="*50)
    print(f"• 총 장소 개수: {len(places)}")

if __name__ == "__main__":
    main() 