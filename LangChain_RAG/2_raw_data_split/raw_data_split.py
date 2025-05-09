'''
2.
    목적: 
        store_raw_data.py로 얻은 raw data를 적절히 분할한다. (이렇게 문서를 작은 조각으로 나누는 이유는 LLM 모델의 입력 토큰의 개수가 정해져 있기 때문)
    메소드: 
        1. 파일 하나에 저장된 장소 개수를 반환한다.
        2. 파일 하나에 저장된 가장 긴 장소 문장의 단어 수를 반환한다.
        3. 파일 하나에 저장된 모든 장소 목록을 반환한다.
    후속 처리: 
        어떤 장소하나를 가져오기 위해서, 메소드 3번 get_all_places를 사용한다.
'''

import boto3
from typing import List, Dict
from io import StringIO
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

class S3:
    def __init__(self, bucket: str = None):
        """
        S3 데이터 처리를 위한 클래스 초기화
        
        Args:
            bucket (str): S3 버킷 이름 (기본값: 환경 변수에서 가져옴)
        """
        self.bucket = bucket or os.getenv('S3_BUCKET_NAME')
        if not self.bucket:
            raise ValueError("S3 버킷 이름이 설정되지 않았습니다. 환경 변수 S3_BUCKET_NAME을 설정하거나 bucket 매개변수를 전달하세요.")
        
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
    

    # 폴더 내 모든 파일 목록 반환
    def list_files_in_folder(self, folder_path: str) -> List[str]:
        """
        S3 버킷의 특정 폴더 내에 있는 모든 파일의 경로를 반환합니다.
        
        Args:
            folder_path (str): 폴더 경로 (예: 'preprocessed_raw_data/')
        
        Returns:
            List[str]: 파일 경로 목록
        """
        # 폴더 경로가 '/'로 끝나지 않으면 추가
        if not folder_path.endswith('/'):
            folder_path += '/'
            
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=folder_path
        )
        
        # 폴더 자체는 제외하고 파일만 반환
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # 폴더 자체는 제외
                if key != folder_path:
                    files.append(key)
                    
        return files


    # 파일 하나 읽기
    def get_file_content(self, file_path: str) -> str:
        """
        S3에서 파일 내용을 읽어옵니다.
        
        Args:
            file_path (str): S3 객체 키
        
        Returns:
            str: 파일 내용
        """
        response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)

        return response['Body'].read().decode('utf-8')
    
    
    # 파일 하나에 저장된 장소 개수 반환
    def count_places(self, file_path: str) -> int:
        """
        S3에 저장된 전처리된 데이터 파일에서 장소 문장 개수를 계산하여 반환합니다.
        
        Args:
            file_path (str): 전처리된 데이터 파일 경로
        
        Returns:
            int: 장소 문장 개수
        """
        content = self.get_file_content(file_path)
        blocks = [bl for bl in content.split('\n\n') if bl.strip()]
        place_blocks = blocks[1:]
        return len(place_blocks)
    
    
    # 파일 하나에 저장된 가장 긴 장소 문장의 단어 수 반환
    def longest_place_word_length(self, file_path: str) -> int:
        """
        S3에 저장된 전처리된 데이터 파일에서 가장 긴 장소 문장의 단어 수를 반환합니다.
        
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


    # 파일 하나에 저장된 모든 장소 목록 반환
    def get_all_places(self, file_path: str) -> List[str]:
        """
        S3에 저장된 전처리된 데이터 파일에서 모든 장소 문장을 리스트로 반환합니다.
        
        Args:
            file_path (str): 전처리된 데이터 파일 경로
        
        Returns:
            List[str]: 장소 문장 리스트
        """
        content = self.get_file_content(file_path)
        blocks = [bl for bl in content.split('\n\n') if bl.strip()]
        # 첫 번째 블록은 메타데이터이므로 제외
        place_blocks = blocks[1:]
        return place_blocks


def main():
    """
    분석 실행 함수
    """
    # S3 인스턴스 생성
    s3 = S3()
    
    # 분석할 파일 경로
    file_path = "preprocessed_raw_data/preprocessed_data_33.119_126.190.txt"
    
    # 폴더 내 모든 파일 목록 출력
    folder_path = "preprocessed_raw_data/"
    files = s3.list_files_in_folder(folder_path)
    print("\n" + "="*50)
    print("폴더 내 파일 목록")
    print("="*50)
    for file in files:
        print(f"• {file}")

    print("\n" + "="*50)
    print(f"파일 분석: {file_path}")
    print("="*50)
    print(s3.get_file_content(file_path))

    print("\n" + "="*50)
    places = s3.get_all_places(file_path)
    print(f"장소 목록 (총 {len(places)}개)")
    print("="*50)
    for i, place in enumerate(places, 1):
        print(f"{i:02d}. {place}\n")
        
    print("="*50)
    print("분석 결과")
    print("="*50)
    total = s3.count_places(file_path)
    longest = s3.longest_place_word_length(file_path)
    print(f"• 총 장소 개수: {total}")
    print(f"• 가장 긴 장소 단어 수: {longest}")
    

if __name__ == "__main__":
    main() 