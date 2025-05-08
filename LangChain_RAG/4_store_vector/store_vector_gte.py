'''
4.
	목적:
		embedded data(vector)를 vector store에 저장한다.
	방법:
		FAISS, Chroma, milvus(추천) 등의 vector store에 저장할 수 있고, 저장되는 방법 또한 여러 방법이 있을 수 있다.
	후속 처리:
		vector store에 저장된 embedded data를 retriver를 사용하여, 최적의 vector를 검색한다.
'''

"""
store_vector.py

목적:
    embedding된 벡터(.npy)를 FAISS 벡터 DB로 저장하고, 메타데이터와 함께 관리한다.
조건:
    - FAISS 사용 (cosine similarity 기반 Index)
    - 메타데이터는 별도 JSON으로 관리
"""

import os
import json
import numpy as np
import faiss

# -----------------------------
# 1. 경로 설정
# -----------------------------
EMBEDDING_DIR = r"C:\Capstone2\LangChain_RAG\embeddings"
FAISS_INDEX_SAVE_PATH = os.path.join(EMBEDDING_DIR, "faiss_index_from_raw.index")
METADATA_SAVE_PATH = os.path.join(EMBEDDING_DIR, "faiss_metadata_from_raw.json")

# 벡터 및 메타데이터 파일 이름 (raw 기준으로 변경 가능)
VECTOR_FILE = "from_raw_json_vectors.npy"
METADATA_FILE = "from_raw_json_metadata.json"

# -----------------------------
# 2. 벡터 및 메타데이터 로드
# -----------------------------
def load_vectors_and_metadata(embedding_dir, vector_file, metadata_file):
    """
    numpy 벡터와 JSON 메타데이터를 로딩하는 함수
    """
    vector_path = os.path.join(embedding_dir, vector_file)
    metadata_path = os.path.join(embedding_dir, metadata_file)

    vectors = np.load(vector_path).astype("float32")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return vectors, metadata

# -----------------------------
# 3. 벡터 정규화 (Cosine Similarity 위해 필요)
# -----------------------------
def normalize_vectors(vectors):
    """
    FAISS에서 cosine similarity를 사용하기 위해 벡터를 L2 정규화한다.
    """
    faiss.normalize_L2(vectors)
    return vectors

# -----------------------------
# 4. FAISS 인덱스 생성 및 저장
# -----------------------------
def create_faiss_index(vectors, index_save_path):
    """
    Cosine similarity 기반의 FAISS 인덱스를 생성하고 저장
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (normalize하면 cosine과 동일)
    index.add(vectors)
    faiss.write_index(index, index_save_path)
    print(f"FAISS 인덱스 저장 완료: {index_save_path}")
    return index

# -----------------------------
# 5. 메타데이터 저장
# -----------------------------
def save_metadata(metadata, save_path):
    """
    벡터 인덱스의 ID와 실제 장소 이름 등의 메타데이터를 함께 저장
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장 완료: {save_path}")

# -----------------------------
# 6. 전체 실행 흐름
# -----------------------------
if __name__ == "__main__":
    print("▶ 벡터 및 메타데이터 로딩 중...")
    vectors, metadata = load_vectors_and_metadata(EMBEDDING_DIR, VECTOR_FILE, METADATA_FILE)

    print("▶ 벡터 정규화 중...")
    normalized_vectors = normalize_vectors(vectors)

    print("▶ FAISS 인덱스 생성 중...")
    index = create_faiss_index(normalized_vectors, FAISS_INDEX_SAVE_PATH)

    print("▶ 메타데이터 저장 중...")
    save_metadata(metadata, METADATA_SAVE_PATH)

    print("\n✅ 벡터 저장 및 인덱싱 완료")
