"""
4.
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import sys
from typing import List
# 프로젝트 루트 디렉토리를 Python 경로에 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # utils의 상위 디렉토리의 상위 디렉토리
sys.path.append(str(PROJECT_ROOT))

# 환경 변수(.env) 로드: 경로 설정 등
load_dotenv()

# 기본 저장 경로: 환경 변수로 재정의 가능
FAISS_INDEX_PATH   = os.getenv("FAISS_INDEX_PATH",   "data/embedding_data/faiss_index.index")
METADATA_SAVE_PATH = os.getenv("METADATA_SAVE_PATH", "data/embedding_data/faiss_metadata.json")


def create_flat_index(vectors: np.ndarray, index_path: str) -> None:
    """
    Flat(Inner Product) 인덱스 생성 및 디스크에 저장
    - vectors: L2 정규화된 임베딩 행렬 (N x D)
    - index_path: 저장할 파일 경로
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # 원본 벡터를 사용한 내적 탐색
    index.add(vectors)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


def create_ivf_index(
    vectors: np.ndarray,
    index_path: str,
    nlist: int = 100,
    nprobe: int = 10
) -> None:
    """
    IVF-Flat 인덱스 생성 및 저장
    - nlist: 코어스 퀀타이저(클러스터) 개수
    - nprobe: 검색 시 탐색할 클러스터 수
    """
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)              # 클러스터 코어스 학습
    index.add(vectors)
    index.nprobe = nprobe             # 검색 시 k개 클러스터 탐색
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


def create_ivfpq_index(
    vectors: np.ndarray,
    index_path: str,
    nlist: int = 100,
    m: int = 8,
    nprobe: int = 10
) -> None:
    """
    IVF-PQ 인덱스 생성 및 저장
    - nlist: 클러스터 수
    - m: 서브 양자 개수 (PQ 파티션)
    - nprobe: 검색 시 탐색할 클러스터 수
    """
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


def create_hnsw_index(
    texts: List[str],
    vectors: np.ndarray,
    m: int = 32,
    efConstruction: int = 40,
    efSearch: int = 16
) -> None:
    """
    HNSW 인덱스 생성 및 저장
    - texts: 각 벡터에 대응하는 원본 텍스트 리스트
    - vectors: (N, D) 형태의 numpy.ndarray (N: 문서 수, D: 임베딩 차원)
    - index_path: 인덱스를 저장할 경로 (예: '/path/to/my_index.index')
    - m: 노드당 연결 수 (기본값: 32)
    - efConstruction: 그래프 빌드 시 후보 집합 크기 (기본값: 40)
    - efSearch: 검색 시 후보 탐색 폭 (기본값: 16)
    """
    # (1) 인덱스 디렉토리 생성
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    # (2) 벡터 차원 수 알아내기
    dim = vectors.shape[1]

    # (3) FAISS HNSW 인덱스 생성
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch

    # (4) 벡터 추가
    index.add(vectors)

    # (5) 인덱스 파일 저장
    faiss.write_index(index, FAISS_INDEX_PATH)

    # (6) 메타데이터(텍스트 리스트)를 JSON으로 함께 저장
    metadata_path = Path(FAISS_INDEX_PATH).with_suffix(".meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] HNSW index 및 메타데이터 저장 완료 → {FAISS_INDEX_PATH}, {metadata_path}")


def save_metadata(texts: list[str], metadata_path: str) -> None:
    """
    임베딩된 각 벡터에 대응하는 원본 텍스트 메타데이터를 JSON으로 저장
    """
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)
