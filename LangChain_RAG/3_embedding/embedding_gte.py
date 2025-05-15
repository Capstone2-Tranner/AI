"""
3.
    입력 :
        S3 버킷의 preprocessed_raw_data/ 폴더에 저장된 분할된 텍스트 파일들
    과정 :
        1. raw_data_split.S3 클래스를 이용해 S3에서 모든 장소 문장을 texts 리스트로 로드
        2. SentenceTransformer 모델(thenlper/gte-base)을 배치(BATCH_SIZE=64) 단위로 호출해 임베딩 벡터(vectors) 생성
        3. sklearn.normalize로 L2 정규화
        4. [{"text":..., "embedding":...}, …] 형태의 JSON 배열로 로컬에 저장 (embeddings.json)
    출력 :
        data/embedding_data/embeddings.json: 각 문장과 그 임베딩이 묶인 JSON
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# raw_data_split.py가 있는 폴더를 추가
BASE_DIR = Path(__file__).resolve().parents[1]          # 프로젝트 루트
SPLIT_DIR = BASE_DIR / "2_raw_data_split"
sys.path.append(str(SPLIT_DIR))

from raw_data_split import S3

load_dotenv()  # AWS_KEY, S3_BUCKET_NAME 등

# ──────────────────────────
# 1. 파라미터 & 모델 로드
# ──────────────────────────
PREFIX        = "preprocessed_raw_data/"
MODEL_NAME    = "thenlper/gte-base"
BATCH_SIZE    = 64
OUTPUT_PATH   = BASE_DIR / "data" / "embedding_data" / "embeddings.json"

s3        = S3()
embedder  = SentenceTransformer(MODEL_NAME)
texts: List[str]   = []
vectors: List[List[float]] = []

# ──────────────────────────
# 2. 텍스트 로드
# ──────────────────────────
for key in s3.list_files_in_folder(PREFIX):
    texts.extend(s3.get_all_places(key))

if not texts:
    raise RuntimeError("S3에서 불러온 문장이 없습니다.")
print(f"[load_texts] 총 {len(texts):,}개의 문장 로드 완료")

# ──────────────────────────
# 3. 임베딩 (배치)
# ──────────────────────────
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    emb = embedder.encode(batch, show_progress_bar=False)
    vectors.extend(emb.tolist())
    print(f"  • {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,} 임베딩 완료")

# L2 정규화
vectors = normalize(np.array(vectors), norm="l2").tolist()
print("[embed] 임베딩 및 정규화 완료")

# ──────────────────────────
# 4. JSON 저장
# ──────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        [{"text": t, "embedding": v} for t, v in zip(texts, vectors)],
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f"[save_json] 저장 완료 → {OUTPUT_PATH}")
