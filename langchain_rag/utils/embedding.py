#!/usr/bin/env python3
# embedding.py
"""
S3(또는 로컬) 전처리 텍스트 → 배치 임베딩 → HNSW 인덱스 추가 → JSONL 메타 기록
--start_entries와 --max_entries 옵션으로 처리 범위를 지정할 수 있습니다.
디버깅용으로 파일 처리 번호와 메모리 로그를 모두 출력합니다.
"""

from __future__ import annotations
import json, os, sys, argparse
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
import psutil

from langchain_community.embeddings import HuggingFaceEmbeddings
from local_storage import LocalStorage

load_dotenv()
HNSW_INDEX_PATH = Path(os.getenv("HNSW_INDEX_PATH", "data/embedding_data/hnsw.index"))
METADATA_PATH   = HNSW_INDEX_PATH.with_suffix(".meta.jsonl")


def log_mem(stage: str):
    m = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[MEMORY] {stage}: {m:.2f} MiB")

class EmbedFromS3:
    def __init__(
        self,
        folder_path: str,
        model_name: str,
        batch_size: int,
        start_entries: int,
        max_entries: int | None,
        m: int,
        ef_construction: int,
        ef_search: int,
    ):
        log_mem("init start")
        self.folder_path   = folder_path
        self.batch_size    = batch_size
        self.start_entries = start_entries
        self.max_entries   = max_entries
        self.localstorage  = LocalStorage()
        self._model        = HuggingFaceEmbeddings(model_name=model_name)
        self._index: faiss.IndexHNSWFlat | None = None
        self._m                = m
        self._ef_construction  = ef_construction
        self._ef_search        = ef_search
        log_mem("init complete")

    def _yield_text_batches(self) -> Iterable[List[str]]:
        file_list = self.localstorage.list_files_in_folder(self.folder_path)
        sent_count = 0

        for idx, fp in enumerate(file_list, start=1):
            if not fp.lower().endswith(".txt"):
      continue
            texts = self.localstorage.get_all_places_from_predata(fp)
            print(f"  • ({idx}/{len(file_list)}) {fp}: {len(texts)}개 문장")

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                sent_count += len(batch)
                # skipping already processed
                if sent_count <= self.start_entries:
                    continue
                # stop if exceeded
                if self.max_entries is not None and sent_count > self.max_entries:
                    return
                yield batch

    def build_hnsw_stream(self) -> None:
        total = 0
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w", encoding="utf-8") as meta_f:
            for batch_texts in self._yield_text_batches():
                log_mem("before embed_documents")
                vecs = self._model.embed_documents(batch_texts)
                log_mem("after embed_documents")

                vecs = normalize(np.asarray(vecs, dtype="float32"), norm="l2")
                log_mem("after normalize")

                if self._index is None:
                    dim = vecs.shape[1]
                    self._index = faiss.IndexHNSWFlat(dim, self._m)
                    self._index.hnsw.efConstruction = self._ef_construction
                    self._index.hnsw.efSearch       = self._ef_search
                    print(f"[INFO] HNSW 초기화: dim={dim}")
                    log_mem("after HNSW init")

                log_mem("before index.add")
                self._index.add(vecs)
                log_mem("after index.add")

                for t in batch_texts:
                    meta_f.write(json.dumps(t, ensure_ascii=False) + "\n")

                total += len(batch_texts)
                log_mem("after cleanup")

        log_mem("before write_index")
        HNSW_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(HNSW_INDEX_PATH))
        log_mem("after write_index")
        print(f"[DONE] 인덱스 및 메타데이터 저장 완료 (총 {total} vectors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path",  default="pre_processed_data/")
    parser.add_argument("--model_name",   default="BM-K/KoSimCSE-roberta")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--start_entries", type=int, default=0, help="건너뛸 문장 수")
    parser.add_argument("--max_entries",   type=int, default=None, help="처리할 최대 문장 수")
    parser.add_argument("--m",               type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=40)
    parser.add_argument("--ef_search",       type=int, default=16)
    args = parser.parse_args()

    EmbedFromS3(
        folder_path=args.folder_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        start_entries=args.start_entries,
        max_entries=args.max_entries,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    ).build_hnsw_stream()

