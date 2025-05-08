'''
2.
    목적: 
        store_raw_data_2_s3_by_multiple_thread.pyd로 얻은 raw data를 적절히 분할한다. (이렇게 문서를 작은 조각으로 나누는 이유는 LLM 모델의 입력 토큰의 개수가 정해져 있기 때문)
    방법: 
        구역 단위(구단위 or 시단위)로 장소 데이터를 split 하거나, 300줄로 split 하거나 여러 경우의 수가 있다.
        이 부분에서 여러 경우로 테스트 해봐서 최적의 split 방법을 골라야한다.
    후속 처리: 
        split된 data(split data)를 embedding.py를 통해 vector로 embedding한다.
'''
"""
이 스크립트는 전처리된 텍스트 파일(preprocessed_data_grid_0012.txt)의
장소 문장 개수와 가장 긴 장소 문장의 단어 수를 계산합니다.
"""

def count_places(
    file_path: str = "Capstone2/LangChain_RAG/data/preprocessed_raw_data/preprocessed_data_grid_0012.txt"
) -> int:
    """
    주어진 전처리된 데이터 파일에서 장소 문장 개수를 계산하여 반환합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    blocks = [bl for bl in content.split('\n\n') if bl.strip()]
    place_blocks = blocks[1:]
    return len(place_blocks)


def longest_place_word_length(
    file_path: str = "Capstone2/LangChain_RAG/data/preprocessed_raw_data/preprocessed_data_grid_0012.txt"
) -> int:
    """
    주어진 전처리된 데이터 파일에서 가장 긴 장소 문장의 단어 수를 반환합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    blocks = [bl for bl in content.split('\n\n') if bl.strip()]
    place_blocks = blocks[1:]
    max_len = 0
    for place in place_blocks:
        word_count = len(place.split())
        if word_count > max_len:
            max_len = word_count
    return max_len


def main():
    """
    분석 실행 함수 (인자 없이 파일 경로를 기본값으로 사용)
    """
    total = count_places()
    longest = longest_place_word_length()
    print(f"총 장소 개수: {total}")
    print(f"가장 긴 장소 단어 수: {longest}")


if __name__ == "__main__":
    main() 