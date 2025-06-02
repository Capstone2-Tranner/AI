"""
3.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
 
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from langchain_community.embeddings import HuggingFaceEmbeddings  # KoSimCSE
# from sentence_transformers import SentenceTransformer #gte

from Capstone2.langchain_rag.utils.local_storage import LocalStorage

# store_vector.py에서 필요한 함수들 import
from Capstone2.langchain_rag.utils.store_vector import create_hnsw_index, save_metadata
 
# 환경 변수 로드
load_dotenv()
 
# 기본 HNSW 인덱스 저장 경로 (환경변수로 재정의 가능)
DEFAULT_HNSW_PATH = Path(
    os.getenv("HNSW_INDEX_PATH", "data/embedding_data/hnsw.index")
)
# 기본 메타데이터 저장 경로
DEFAULT_META_PATH = Path(
    os.getenv("METADATA_SAVE_PATH", "data/embedding_data/faiss_metadata.json")
)

class EmbedFromS3:
    def __init__(
        self,
        folder_path: str = "test/raw_data/",        # S3 폴더 경로
        model_name: str = "BM-K/KoSimCSE-roberta", # 임베딩 모델명
        # model_name: str = "thenlper/gte-base", 
        batch_size: int = 64                   # 배치 크기
    ):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self._localstorage = LocalStorage()
        self._model = HuggingFaceEmbeddings(model_name)
        # self._model = SentenceTransformer(model_name)

    # ──────────────────────────
    # 임베딩된 내용 vector store에 저장
    # ──────────────────────────   
    def _save_to_vector_store(self, texts: List[str], vectors: np.ndarray) -> None:
            """
            벡터와 메타데이터를 S3에 저장
            """
            self._localstorage.save_vector_store(
                texts=texts,
                vectors=vectors,
                base_path="embedding"
            )
    
    # ──────────────────────────
    # raw 데이터 임베딩
    # ──────────────────────────
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        주어진 텍스트 리스트를 임베딩한 뒤 L2 정규화하여 반환
        """
        embs = self._model.embed_documents(texts)
        # embs = self._model.encode(texts, show_progress_bar=False) # gte (sentence_transformers)
        return normalize(np.asarray(embs, dtype="float32"), norm="l2")
 
    # ──────────────────────────
    # 1. 단일 파일 임베딩
    # ──────────────────────────
    def embed_file(self, file_path: str) -> Tuple[List[str], np.ndarray]:
        """
        1) 단일 JSON/TXT 파일에서 텍스트 추출
        2) 임베딩 벡터 생성 + 정규화
        3) HNSW 인덱스와 메타데이터 저장
        """
        # 1) 텍스트 읽기
        texts = (
            self._localstorage.get_all_places_from_json(file_path)
            if file_path.lower().endswith(".json")
            else self._localstorage.get_all_places_from_txt(file_path)
        )
        # 2) 임베딩
        vectors = self._embed_texts(texts)
 
        # 3) HNSW 인덱스 저장
        self._save_to_vector_store(texts, vectors)   
 
        return texts, vectors
 
    # ──────────────────────────
    # 2. 앞에서 k개 파일 임베딩
    # ──────────────────────────
    def embed_k_files(self, k: int) -> Tuple[List[str], np.ndarray]:
        """
        1) S3에서 첫 k개 파일 가져오기
        2) 순차 임베딩 + 정규화
        3) HNSW 인덱스와 메타데이터 저장
        """
        keys = self._localstorage.list_first_n_files(self.folder_path, k)
        all_texts: List[str] = []
        print(f"[DEBUG] embed_k_files 시작: 처리할 파일 수 = {len(keys)}")
        for idx, key in enumerate(keys, start=1):
            print(f"[DEBUG] ({idx}/{len(keys)}) 파일 읽는 중: {key}")
            # 파일별 텍스트 추출
            texts = self._localstorage.get_all_places_from_json(key)
            print(f"[DEBUG]   -> 추출된 텍스트 개수: {len(texts)}")
            all_texts.extend(texts)

        print(f"[DEBUG] 전체 텍스트 개수: {len(all_texts)} — 임베딩 시작")
        # 임베딩
        vectors = self._embed_texts(all_texts)
        print(f"[DEBUG] 임베딩 완료: 벡터 shape = {vectors.shape}")

        # HNSW 인덱스 저장
        self._save_to_vector_store(all_texts, vectors)
        print(f"[DEBUG] 인덱스 및 메타데이터 저장 완료")

        return all_texts, vectors
 
    # ──────────────────────────
    # 3. 전체 폴더 임베딩
    # ──────────────────────────
    def embed_all(self) -> Tuple[List[str], np.ndarray]:
        """
        1) S3 폴더 내 모든 파일 목록 조회
        2) 파일별로 배치 단위 임베딩 + 정규화
        3) HNSW 인덱스와 메타데이터 저장
        """
        file_list = self._localstorage.list_files_in_folder(self.folder_path)
        all_texts: List[str] = []
        print(f"[DEBUG] embed_all 시작: 폴더 내 파일 수 = {len(file_list)}")
        for idx, fp in enumerate(file_list, start=1):
            print(f"[DEBUG] ({idx}/{len(file_list)}) 파일 읽는 중: {fp}")
            # 전체 파일 텍스트 추출
            if fp.lower().endswith(".json"):
                texts = self._localstorage.get_all_places_from_json(fp)
            else:
                texts = self._localstorage.get_all_places_from_txt(fp)
            print(f"[DEBUG]   -> 추출된 텍스트 개수: {len(texts)}")
            all_texts.extend(texts)

        print(f"[DEBUG] 전체 텍스트 개수: {len(all_texts)} — 임베딩 시작")
        # 임베딩
        vectors = self._embed_texts(all_texts)
        print(f"[DEBUG] 임베딩 완료: 벡터 shape = {vectors.shape}")

        # HNSW 인덱스 & 메타데이터 저장
        self._save_to_vector_store(all_texts, vectors)
        print(f"[DEBUG] 인덱스 및 메타데이터 저장 완료")

        return all_texts, vectors
 
# ────────────────────────
# CLI 테스트
# ────────────────────────
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="단일 S3 키 지정")
    group.add_argument("--n", type=int, metavar="N", help="첫 N개 파일 처리")
    group.add_argument("--all", action="store_true", help="폴더 전체 처리")
    args = parser.parse_args()
 
    emb = EmbedFromS3()
    if args.file:
        texts, vecs = emb.embed_file(args.file)
    elif args.n is not None:
        texts, vecs = emb.embed_k_files(args.n)
    else:
        texts, vecs = emb.embed_all()
 
    print(f" 임베딩 및 저장 완료 • 총 {len(texts)}개 • shape={vecs.shape}")
 
