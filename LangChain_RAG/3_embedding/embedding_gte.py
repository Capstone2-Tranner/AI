'''
3.
    목적:
        split된 data(split data)를 vector로 embedding된다.
    방법:
        OpenAIEmbedding, huggingface embedding 등 여러 가지 방법이 있다.
    후속 처리:
        embedded data를 vector store에 저장한다.
'''

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# [1] 파일 경로 설정
# -----------------------------
# 원본 JSON 파일 경로
RAW_JSON_PATH = r"C:\Users\alsdn\바탕 화면\Capstone2\LangChain_RAG\data\raw_data\raw_data_grid_0012.json"

# 전처리된 텍스트 파일 경로 (한 줄에 하나의 장소 정보가 문장으로 저장된 형식)
PREPROCESSED_TXT_PATH = r"C:\Users\alsdn\바탕 화면\Capstone2\LangChain_RAG\data\preprocessed_raw_data\preprocessed_data_grid_0012.txt"

# 임베딩 결과 저장 디렉터리
OUTPUT_DIR = r"C:\Users\alsdn\바탕 화면\Capstone2\LangChain_RAG\embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# [2] HuggingFace 임베딩 모델 로딩
# -----------------------------
# thenlper/gte-base: GTE(Gaussian Transformer Encoder) 모델, 다국어 성능 우수
model = SentenceTransformer("thenlper/gte-base")

# -----------------------------
# [3] 임베딩과 결과 저장 함수 정의
# -----------------------------
def embed_texts_and_save(texts, place_names, prefix):
    """
    주어진 텍스트 리스트를 임베딩하고, 결과 벡터와 메타데이터를 저장한다.
    - texts: 임베딩할 텍스트 리스트
    - place_names: 각 텍스트에 해당하는 장소 이름
    - prefix: 저장할 파일 이름에 사용할 접두어 (e.g., 'from_raw_json')
    """
    vectors = model.encode(texts, show_progress_bar=True)

    # 벡터 저장 (.npy 형식)
    vector_path = os.path.join(OUTPUT_DIR, f"{prefix}_vectors.npy")
    np.save(vector_path, vectors)

    # 메타데이터 저장 (index와 place_name 매핑)
    metadata_path = os.path.join(OUTPUT_DIR, f"{prefix}_metadata.json")
    metadata = [{"index": i, "place_name": place_names[i]} for i in range(len(place_names))]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[{prefix}] 벡터 저장 완료: {vector_path}")
    print(f"[{prefix}] 메타데이터 저장 완료: {metadata_path}")

# -----------------------------
# [4] 원본 JSON 파일 처리 함수
# -----------------------------
def process_raw_json(file_path):
    """
    원본 JSON 데이터를 불러와 텍스트로 변환하고, 임베딩할 문장 리스트와 장소 이름 리스트를 반환한다.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # JSON이 dict 구조일 경우, 내부 key를 기준으로 리스트 추출
    if isinstance(raw_data, dict):
        for key in ["places", "data", "results"]:
            if key in raw_data and isinstance(raw_data[key], list):
                raw_data = raw_data[key]
                break
        else:
            raise ValueError("알 수 없는 딕셔너리 구조입니다. 리스트로 된 장소 배열을 포함하고 있어야 합니다.")
    elif not isinstance(raw_data, list):
        raise ValueError("지원하지 않는 JSON 최상위 구조입니다.")

    # 내부가 문자열 리스트일 경우, 문자열을 다시 JSON으로 파싱
    if isinstance(raw_data[0], str):
        raw_data = [json.loads(item) for item in raw_data]

    texts, names = [], []

    for item in raw_data:
        name = item.get("name", "")
        types = ", ".join(item.get("types", []))
        rating = item.get("rating", "정보 없음")

        price_level = item.get("price_level", None)
        price_desc = (
            "저렴한 편" if price_level == 1 else
            "중간 가격대" if price_level == 2 else
            "비싼 편" if price_level == 3 else
            "정보 없음"
        )

        opening_hours = item.get("opening_hours", {}).get("weekday_text", [])
        opening_str = ", ".join(opening_hours) if opening_hours else "정보 없음"

        reviews = item.get("reviews", [])
        review_texts = [
            f"'{r.get('text', '')}' (평점: {r.get('rating', '')})"
            for r in reviews[:5]
        ]
        review_str = ", ".join(review_texts) if review_texts else "리뷰 없음"

        # 최종 문장 구성
        full_text = (
            f"{name}은(는) 대한민국 제주특별자치도 서귀포시 대정읍 가파리에 위치한 "
            f"{types}입니다. 평점은 {rating}점입니다. 가격 수준은 {price_desc}입니다. "
            f"운영 시간은 {opening_str}입니다. 리뷰는 다음과 같습니다: {review_str}"
        )

        texts.append(full_text)
        names.append(name)

    return texts, names

# -----------------------------
# [5] 전처리된 텍스트(.txt) 파일 처리 함수
# -----------------------------
def process_preprocessed_txt(file_path):
    """
    전처리된 텍스트 파일을 불러와 문장 리스트와 장소 이름 리스트를 생성한다.
    - 장소 이름은 "은(는)" 이전 텍스트로 자동 추출된다.
    """
    texts, names = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "은(는)" in line:
                place_name = line.split("은(는)")[0].strip()
            else:
                place_name = "unknown"
            texts.append(line)
            names.append(place_name)
    return texts, names

# -----------------------------
# [6] 메인 실행 블록
# -----------------------------
if __name__ == "__main__":
    print("원본 JSON 처리 중...")
    raw_texts, raw_names = process_raw_json(RAW_JSON_PATH)
    embed_texts_and_save(raw_texts, raw_names, prefix="from_raw_json")

    print("\n 전처리된 TXT 처리 중...")
    pre_texts, pre_names = process_preprocessed_txt(PREPROCESSED_TXT_PATH)
    embed_texts_and_save(pre_texts, pre_names, prefix="from_preprocessed_txt")

    print("\n 모든 임베딩 완료")
