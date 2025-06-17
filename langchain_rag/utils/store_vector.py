#!/usr/bin/env python3
# store_vector.py
"""
Faiss 인덱스 생성 및 메타데이터 저장 함수 모음
- create_ivfpq_index: IVF-PQ 인덱스 생성
- create_hnsw_index: HNSW 인덱스 생성
- save_metadata      : 메타데이터 저장
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

import faiss
import numpy as np

load_dotenv()

FAISS_INDEX_PATH   = Path(os.getenv("FAISS_INDEX_PATH",   "data/embedding_data/faiss_index.index"))
METADATA_SAVE_PATH = Path(os.getenv("METADATA_SAVE_PATH", "data/embedding_data/faiss_metadata.json"))


def create_ivfpq_index(
    vectors: list[np.ndarray] | np.ndarray,
    dim: int,
    nlist: int = 100,
    m: int = 8,
    nprobe: int = 10,
    max_batches: int | None = None,
) -> None:
    """
    IVF-PQ 인덱스를 배치 단위로 생성 및 저장합니다.
    - vectors: (N,D) ndarray 혹은 [(B,D), ...] 리스트
    - dim    : 벡터 차원
    - max_batches: 처리할 배치 수 제한 (없으면 전체)
    """
    # 1) 코어스 퀀타이저 학습
    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    # 2) 샘플로 학습
    if isinstance(vectors, list):
        sample = np.vstack(vectors[: min(len(vectors), 50_000)])
    else:
        sample = vectors[:50_000]
    index.train(sample)

    # 3) 배치별로 추가
    if isinstance(vectors, list):
        for i, batch in enumerate(vectors, 1):
            if max_batches is not None and i > max_batches:
                break
            index.add(batch)
    else:
        index.add(vectors)

 # 4) 인덱스 저장
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def create_hnsw_index(
    texts: list[str],
    vectors: np.ndarray | list[np.ndarray],
    m: int = 32,
    efConstruction: int = 40,
    efSearch: int = 16,
    max_batches: int | None = None,
) -> None:
    """
    HNSW 인덱스를 배치 단위로 생성 및 저장합니다.
    - texts  : 원본 텍스트 리스트 (메타데이터용)
    - vectors: (N,D) ndarray 혹은 [(B,D), ...] 리스트
    - max_batches: 처리할 배치 수 제한 (없으면 전체)
    """
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 벡터 반복자 준비
    vec_iter = vectors if isinstance(vectors, list) else [vectors]
    dim      = vec_iter[0].shape[1]

    # HNSW 초기화
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch       = efSearch

    # 배치별로 추가
    for i, batch in enumerate(vec_iter, 1):
        if max_batches is not None and i > max_batches:
            break
        index.add(batch)

    # 인덱스 저장
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def save_metadata(
    texts: list[str],
    metadata_path: str | None = None,
) -> None:
    """
    임베딩된 벡터에 대응하는 텍스트 메타데이터를 JSON으로 저장합니다.
    - metadata_path: 경로 지정이 없으면 .env 기반 METADATA_SAVE_PATH 사용
    """
    path = Path(metadata_path) if metadata_path else METADATA_SAVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 메타데이터 저장 → {path}")
