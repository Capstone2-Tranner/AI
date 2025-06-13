"""
3.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Tuple
 
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from langchain_community.embeddings import HuggingFaceEmbeddings  # KoSimCSE
# from sentence_transformers import SentenceTransformer #gte

from local_storage import LocalStorage

# 프로젝트 루트 디렉토리를 Python 경로에 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # utils의 상위 디렉토리의 상위 디렉토리
sys.path.append(str(PROJECT_ROOT))

# store_vector.py에서 필요한 함수들 import
from store_vector import create_hnsw_index, save_metadata
 
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
        folder_path: str = "pre_processed_data/",        # S3 폴더 경로
        model_name: str = "BM-K/KoSimCSE-roberta", # 임베딩 모델명
        # model_name: str = "thenlper/gte-base", 
        batch_size: int = 64                   # 배치 크기
    ):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.localstorage = LocalStorage()
        self._model = HuggingFaceEmbeddings(model_name = model_name)
        # self._model = SentenceTransformer(model_name)

    # ──────────────────────────
    # 임베딩된 내용 vector store에 저장
    # ──────────────────────────   
    def _save_to_vector_store(self, texts: List[str], vectors: np.ndarray) -> None:
            """
            벡터와 메타데이터를 vector_db에 저장
            """
            create_hnsw_index(
                texts=texts,
                vectors=vectors
            )
    
    # ───────────────────────────────────────────────
    # 수정 ① : _embed_texts() 를 배치 기반으로 분할 호출
    # ───────────────────────────────────────────────
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """texts 를 self.batch_size 로 잘라가며 임베딩 + L2 정규화"""
        vec_chunks = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            embs  = self._model.embed_documents(chunk)           # (B, dim)
            vec_chunks.append(embs)
        full = np.vstack(vec_chunks).astype("float32")
        return normalize(full, norm="l2")

    # ───────────────────────────────────────────────
    # 수정 ② : embed_all() – 배치로 읽되, 벡터는 한 번만 생성
    # ───────────────────────────────────────────────
    def embed_all(self) -> Tuple[List[str], np.ndarray]:
        file_list = self.localstorage.list_files_in_folder(self.folder_path)
        print(f"[DEBUG] embed_all 시작: {len(file_list)}개 파일")

        all_texts: List[str] = []
        for idx, fp in enumerate(file_list, 1):
            if not fp.lower().endswith(".txt"):
                continue
            texts = self.localstorage.get_all_places_from_predata(fp)
            print(f"[DEBUG] ({idx}/{len(file_list)}) {fp}: {len(texts)}개 추출")
            all_texts.extend(texts)

        print(f"[DEBUG] 전체 텍스트 {len(all_texts)}개 — 배치 임베딩 시작")
        vectors = self._embed_texts(all_texts)      # 내부에서 배치 처리
        print(f"[DEBUG] 임베딩 완료: shape={vectors.shape}")

        self._save_to_vector_store(all_texts, vectors)
        print("[DEBUG] create_hnsw_index() 호출 완료")

        return all_texts, vectors

    
# ────────────────────────
# CLI 테스트
# ────────────────────────
if __name__ == "__main__":
    import argparse
 
    # parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--file", help="단일 S3 키 지정")
    # group.add_argument("--n", type=int, metavar="N", help="첫 N개 파일 처리")
    # group.add_argument("--all", action="store_true", help="폴더 전체 처리")
    # args = parser.parse_args()
 
    emb = EmbedFromS3()
    texts, vecs = emb.embed_all()
    # if args.file:
    #     texts, vecs = emb.embed_file(args.file)
    # elif args.n is not None:
    #     texts, vecs = emb.embed_k_files(args.n)
    # else:
        # texts, vecs = emb.embed_all()
 
    print(f" 임베딩 및 저장 완료 • 총 {len(texts)}개 • shape={vecs.shape}")