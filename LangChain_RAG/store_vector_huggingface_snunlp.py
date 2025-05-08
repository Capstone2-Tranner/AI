'''
4.
	목적:
		embedded data(vector)를 vector store에 저장한다.
	방법:
		FAISS, Chroma, milvus(추천) 등의 vector store에 저장할 수 있고, 저장되는 방법 또한 여러 방법이 있을 수 있다.
	후속 처리:
		vector store에 저장된 embedded data를 retriver를 사용하여, 최적의 vector를 검색한다.
'''
import os
import json
from pathlib import Path

import faiss
import numpy as np


def load_embeddings(emb_file: Path) -> tuple[list[str], np.ndarray]:
    """
    embeddings.json 파일을 읽어 텍스트 리스트와 임베딩 배열을 반환합니다.
    """
    with emb_file.open("r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [rec["text"] for rec in records]
    vectors = np.array([rec["embedding"] for rec in records], dtype="float32")
    return texts, vectors


def main():
    # 파일 경로 설정
    base_dir = Path(__file__).parent
    emb_file = base_dir / "data" / "embedding_data" / "embeddings.json"

    # 임베딩 로드
    texts, embeddings = load_embeddings(emb_file)
    if embeddings.size == 0:
        print("임베딩 데이터가 없습니다. 먼저 embedding.py를 실행하세요.")
        return

    # FAISS 인덱스 생성 (L2 거리)
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    # 2) Inner-Product 인덱스 생성
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

#여기서는 저장 방식을 그냥 순차적으로 보관해서 retrieval시에 모든 벡터와 L2 거리를 계산해서 가까운 벡터들을 순서대로 반환함
#Flat L2 -> 모든 벡터를 순차적으로 구조없이 메모리에 저장하고, 검색 시 모든 벡터와의 거리를 계산함(정확하지만 속도가 느림)
#IndexIVFFlat -> 클러스터링 기반(클러스터링 학습이 필요하나 속도가 빠름\름)
#IndexIVFPQ / OPQ -> 클러스터링 + 양자화 기반(벡터를 저비트 코드로 압축 -> 정확도 저하 가능성)
#IndexHNSWFlat -> 그래프 기반(최근접 이웃 탐색)(계층적인 저장 방식으로 검색 속도 빠르지만 메모리 사용량 높음)
#IndexIVFScalarQuantizer -> 클러스터링 + 스칼라 양자화(벡터를 저비트 코드로 압축 -> 정확도 저하 가능성, OPQ보다는 덜 압축함함)

# | 인덱스         | 검색 속도        | 정확도(Recall) | 메모리 사용 | 동적 업데이트 | 구축 복잡도     |
# | -------------- | ------------    | -----------    | ------     | -------      | -------------   |
# | Flat           | 느림 (O(N))      | 100%          | 높음        | 지원 안 함   | 매우 간단        |
# | IVFFlat        | 빠름 (O(N/n))    | 중·상         | 중간        | 지원 안 함   | 간단 (train 필요)|
# | IVFPQ          | 매우 빠름        | 중            | 매우 낮음   | 지원 안 함   | 복잡 (PQ train)  |
# | HNSWFlat       | 매우 빠름 (logN) | 상(≈100%)     | 높음        | 지원         | 중간            |
# | IVFScalarQuant | 빠름             | 중            | 낮음        | 지원 안 함   | 간단            |

    # 저장할 디렉토리 생성
    vector_db_dir = base_dir / "data" / "vector_db"
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    # 인덱스 파일 경로
    index_path = vector_db_dir / "faiss_index.idx"

    # Windows에서 한글 경로 이슈 회피: Python I/O를 이용해 직렬화 저장
    index_bytes = faiss.serialize_index(index)
    with open(index_path, "wb") as f:
        f.write(index_bytes)

    # 텍스트 매핑 저장
    texts_path = vector_db_dir / "texts.json"
    with texts_path.open("w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"FAISS 인덱스를 '{index_path}'에 저장했습니다.")
    print(f"텍스트 매핑을 '{texts_path}'에 저장했습니다.")


if __name__ == '__main__':
    main()

