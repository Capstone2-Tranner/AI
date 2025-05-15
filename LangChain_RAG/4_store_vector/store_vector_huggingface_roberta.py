#!/usr/bin/env python3
# =============================================================================
#  LangChain_RAG / 4_store_vector / vector_db / vector_store_builder.py
#  ---------------------------------------------------------------------------
#  ■ 목적(Purpose)
#     ➊ S3 버킷 → KoSimCSE 임베딩 → N×D 벡터 행렬을 생성하고
#     ➋ FAISS-HNSW 인덱스(IndexHNSWFlat)로 빌드한 뒤
#     ➌ 인덱스 바이너리(.idx) + 텍스트 매핑(texts.json)을 디스크에 저장한다.
#
#  ■ 클래스 구조
#     • VectorStoreBuilder.run()           : 파이프라인 전체(임베딩→인덱스→저장)
#     • VectorStoreBuilder._generate_embeddings() : S3 임베딩 (EmbedFromS3 재사용)
#     • VectorStoreBuilder._build_index()  : FAISS-HNSW 인덱스 생성
#     • VectorStoreBuilder._save_files()   : 결과물 저장(.idx, .json)
#
#  ■ 저장 디렉터리
#     data/vector_db/
#        ├─ faiss_index_hnsw.idx   ← 벡터 인덱스
#        └─ texts.json             ← 인덱스와 동일 순서의 원본 문장 리스트
#
#  ■ 실행 예시
#     $ python vector_db/vector_store_builder.py
#     → S3에서 임베딩을 만들고, 인덱스를 저장한 뒤 경로를 출력
# =============================================================================
from __future__ import annotations            # PEP-563: postpone evaluation

# ── 표준 라이브러리 ───────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ── 서드파티 라이브러리 ────────────────────────────────────────────────────────
import faiss                                  # Facebook AI Similarity Search
import numpy as np

# ── 내부 모듈: S3 임베딩 파이프라인 ────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]  # LangChain_RAG/
EMBED_DIR     = BASE_DIR / "3_embedding"             # EmbedFromS3 모듈 위치
sys.path.append(str(EMBED_DIR))                      # import 우회

from embedding_huggingface_roberta import EmbedFromS3  # KoSimCSE 임베더 wrapper


# =============================================================================
#  VectorStoreBuilder
# =============================================================================
class VectorStoreBuilder:
    """
    S3 원본 데이터를 KoSimCSE로 임베딩하고,
    FAISS IndexHNSWFlat(=HNSW + L2) 인덱스로 변환해 디스크에 저장한다.

    Parameters
    ----------
    db_dir : str | Path
        인덱스(.idx)·텍스트(.json) 저장 경로(폴더)
    hnsw_m : int
        HNSW 그래프의 최대 이웃 수 (M). 클수록 정확도↑, 메모리/속도↓.
    ef_c  : int
        인덱스 구축 시 efConstruction 값(검색 정확도 목표치).
    ef_s  : int
        검색 시 efSearch 기본값(탐색 폭). load 후 변경 가능.
    """

    # --------------------------------------------------------------------- #
    # 1) 생성자: 파일 경로 및 HNSW 파라미터 초기화
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        db_dir: Path | str = "data/vector_db", # 메타데이터 저장 경로
        hnsw_m: int = 32,         # 일반적으로 16~48 범위 
        ef_c: int = 200,          # 인덱스 생성 정확도 파라미터
        ef_s: int = 50,           # 검색 정확도 파라미터
    ) -> None:
        # ▸ 저장 디렉터리 및 출력 파일 경로
        self.db_dir      = Path(db_dir)
        self.index_path  = self.db_dir / "faiss_index_hnsw.idx"
        self.texts_path  = self.db_dir / "texts.json"

        # ▸ HNSW 파라미터 보관
        self.hnsw_m = hnsw_m
        self.ef_c   = ef_c
        self.ef_s   = ef_s

        # ▸ 내부 상태 변수 (임베딩·인덱스)
        self.texts:   List[str]            = []            # 문장 리스트
        self.vectors: np.ndarray           = np.empty((0, 0), dtype="float32")
        self.index:   Optional[faiss.Index] = None         # FAISS 인덱스 객체

    # --------------------------------------------------------------------- #
    # 2) S3 → KoSimCSE 임베딩
    # --------------------------------------------------------------------- #
    def _generate_embeddings(self, batch_size: int = 64) -> None:
        """
        EmbedFromS3 를 호출해
        • self.texts  : List[str]  (원본 문장)
        • self.vectors: np.ndarray (N, D)  (L2 정규화된 임베딩)
        를 채운다.
        """
        print("[VectorStoreBuilder] S3 임베딩을 생성합니다…")
        self.texts, self.vectors = EmbedFromS3(batch_size=batch_size).run(save_json=False)

    # --------------------------------------------------------------------- #
    # 3) FAISS-HNSW 인덱스 빌드
    # --------------------------------------------------------------------- #
    def _build_index(self) -> None:
        """
        numpy 벡터 행렬 → faiss.IndexHNSWFlat 으로 변환
        • L2 거리(L2flat) + HNSW 그래프 구조
        • efConstruction / efSearch 파라미터 튜닝
        """
        if self.vectors.size == 0:
            raise ValueError("벡터가 없습니다. 임베딩을 먼저 로드/생성하세요.")

        dim   = self.vectors.shape[1]                      # 벡터 차원(D)
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m)      # HNSWFlat(L2)
        index.hnsw.efConstruction = self.ef_c              # 빌드 정확도
        index.hnsw.efSearch       = self.ef_s              # 기본 탐색 폭
        index.add(self.vectors)                            # 벡터 등록
        self.index = index
        print("[VectorStoreBuilder] FAISS-HNSW 인덱스 구축 완료")

    # --------------------------------------------------------------------- #
    # 4) 결과 파일 저장 (.idx, .json)
    # --------------------------------------------------------------------- #
    def _save_files(self) -> None:
        """
        ➊ faiss.write_index() 로 바이너리 저장
        ➋ texts.json  에는 문장 배열을 그대로 dump(indent=2)
        """
        if self.index is None:
            raise ValueError("index 가 없습니다. _build_index() 먼저 수행하세요.")

        # 폴더 생성(존재시 무시)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))
        with self.texts_path.open("w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        print(f"✅ 인덱스   → {self.index_path}")
        print(f"✅ 텍스트   → {self.texts_path}")

    # --------------------------------------------------------------------- #
    # 5) 전체 파이프라인 실행
    # --------------------------------------------------------------------- #
    def run(self, batch_size: int = 64) -> Tuple[Path, Path]:
        """
        1) S3 임베딩 → 2) 인덱스 빌드 → 3) 저장 을 연속 수행

        Returns
        -------
        Tuple[Path, Path]
            (faiss_index_path, texts_json_path)
        """
        self._generate_embeddings(batch_size)
        self._build_index()
        self._save_files()
        return self.index_path, self.texts_path


# =============================================================================
# 6. CLI 진입점
# =============================================================================
if __name__ == "__main__":
    # 필요한 경우 batch_size, HNSW 파라미터를 수정 후 실행
    VectorStoreBuilder().run(batch_size=64)

