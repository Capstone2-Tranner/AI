'''
5.
    목적:
        vector store에서 유사한 벡터를 검색하고 유사도를 계산한다.
    방법:
        vector store에서 쿼리와 유사한 벡터를 검색하고, 유사도 점수를 계산한다.
        검색 결과는 유사도에 따라 정렬된다.
    후속 처리:
        검색된 결과는 make_prompt에서 프롬프트 생성에 사용된다.
'''
"""
retrieval.py

목적:
    사용자의 쿼리를 벡터로 임베딩하고, FAISS 인덱스를 통해 유사한 장소를 검색한다.
조건:
    - 저장된 FAISS 인덱스(.index)와 메타데이터(.json)를 로드
    - cosine similarity 기반 검색
    - SentenceTransformer("thenlper/gte-base") 사용
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. 경로 설정
# -----------------------------
EMBEDDING_DIR = r"C:\Capstone2\LangChain_RAG\embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDING_DIR, "faiss_index_from_raw.index")
METADATA_PATH = os.path.join(EMBEDDING_DIR, "faiss_metadata_from_raw.json")

# -----------------------------
# 2. 모델 및 인덱스 로드
# -----------------------------
print("▶ 임베딩 모델 및 인덱스 로드 중...")
model = SentenceTransformer("thenlper/gte-base")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -----------------------------
# 3. 검색 함수 정의
# -----------------------------
def search_similar_places(query: str, top_k: int = 5):
    """
    사용자 쿼리를 임베딩한 후 FAISS를 통해 유사 장소 Top-k 반환
    """
    query_vec = model.encode([query], normalize_embeddings=True)  # cosine similarity 위해 normalize
    D, I = index.search(query_vec, top_k)  # D: 유사도, I: 인덱스

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(metadata):
            results.append({
                "place_name": metadata[idx]["place_name"],
                "score": float(score)
            })

    return results

# -----------------------------
# 4. 예시 실행
# -----------------------------
if __name__ == "__main__":
    print("🔎 장소 검색 예시")
    while True:
        user_query = input("\n검색할 내용을 입력하세요 (예: '경치 좋은 공원', '짜장면 맛집') → ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break

        top_results = search_similar_places(user_query, top_k=5)
        print("\n🔍 유사한 장소 Top 5:")
        for i, result in enumerate(top_results):
            print(f"{i+1}. {result['place_name']} (유사도: {result['score']:.4f})")
