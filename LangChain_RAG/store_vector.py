# store_vector_gte.py
# ─────────────────────────────────────────────
# FAISS 인덱스 생성 및 임베딩 메타데이터 저장 유틸
# • create_flat_index      : 정확도 100% Flat(Inner Product) 인덱스 생성
# • create_ivf_index       : IVF-Flat 클러스터 인덱스 생성
# • create_ivfpq_index     : IVF + Product Quantization 인덱스 생성
# • create_hnsw_index      : HNSW 그래프 기반 근사 인덱스 생성
# • save_metadata          : 임베딩 텍스트 메타데이터 JSON 저장
# ─────────────────────────────────────────────

import os
import json
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import sys, json
 

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
    vectors: np.ndarray,
    index_path: str,
    m: int = 32,
    efConstruction: int = 40,
    efSearch: int = 16
) -> None:
    """
    HNSW 인덱스 생성 및 저장
    - m: 노드당 연결 수
    - efConstruction: 그래프 빌드 시 후보 집합 크기
    - efSearch: 검색 시 후보 탐색 폭
    """
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(vectors)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


def save_metadata(texts: list[str], metadata_path: str) -> None:
    """
    임베딩된 각 벡터에 대응하는 원본 텍스트 메타데이터를 JSON으로 저장
    """
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)
