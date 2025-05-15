#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────────
#  ⬛  LangChain_RAG / 3_embedding / embed_from_s3.py
#      ──────────────────────────────────────────────────────────────────────────
#  ✓ 역할
#    1. S3 버킷에서 raw JSON / txt 파일을 모두 스캔한 뒤
#    2. 장소 정보를 문자열(text) 리스트로 수집하고
#    3. KoSimCSE-roberta 모델로 임베딩 → L2 정규화
#    4. (선택) texts + vectors 를 JSON 파일로 저장
#
#  ✓ 전제
#    · `2_raw_data_split/json_data_split.py` 에 S3 유틸(S3 클래스)이 정의돼 있어야 함
#    · .env 파일에 AWS 자격증명 & S3_BUCKET_NAME & AWS_REGION 정의
#    · HuggingFace 모델은 인터넷 또는 로컬 캐시에서 자동 다운로드
#
#  ✓ 실행
#      $ python embedding/embed_from_s3.py
# ------------------------------------------------------------------------------
from __future__ import annotations                    # 파이썬 3.11 이전 버전 호환
import sys, json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv                         # .env 로더
from sklearn.preprocessing import normalize            # L2 정규화
from langchain_community.embeddings import HuggingFaceEmbeddings

# ────────────────────────────────────────────────────────────────────────────────
# 0. S3 유틸 모듈 import 준비
#    • 프로젝트 트리 구조를 기준으로 상위 폴더(=repo root) → 2_raw_data_split 아래
#      json_data_split.py 파일의 S3 클래스를 import할 수 있도록 sys.path 에 추가
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]     # LangChain_RAG/
RAW_SPLIT_DIR = BASE_DIR / "2_raw_data_split"
sys.path.append(str(RAW_SPLIT_DIR))                     # import 우회 경로 등록

from raw_data_split import S3                          # ✅ JSON·TXT 파서가 포함된 S3

# .env 파일을 읽어서 환경 변수로 주입 (AWS_ACCESS_KEY_ID 등)
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# 1. 임베딩 파이프라인 클래스
# ────────────────────────────────────────────────────────────────────────────────
class EmbedFromS3:
    """
    S3 버킷에 저장된 raw JSON(.json) & 전처리 TXT(.txt) 파일을
    KoSimCSE-roberta 모델로 임베딩해 주는 헬퍼 클래스
    """

    # ──────────────────────────────
    # 1-1. 생성자
    # ──────────────────────────────
    def __init__(
        self,
        prefix: str = "raw_data/",                      # S3 폴더 prefix(접두사)
        model_name: str = "BM-K/KoSimCSE-roberta",      # HuggingFace 모델 경로
        batch_size: int = 64,                           # 임베딩 배치 크기
        output_path: Path | None = None,                # 저장 경로(생략 시 기본값)
    ):
        # 인스턴스 변수 세팅
        self.prefix      = prefix
        self.batch_size  = batch_size
        self.output_path = output_path or (
            BASE_DIR / "data" / "embedding_data" / "embeddings.json"
        )

        # S3 클라이언트(S3 유틸)와 임베딩 모델 초기화
        self._s3       = S3()                           # 앞서 구현한 S3 클래스
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

        # 결과 저장용 컨테이너
        self.texts:   List[str]              = []       # 원본 문장 리스트
        self.vectors: np.ndarray | List[List[float]] = [] # 임베딩 결과

    # ──────────────────────────────
    # 1-2. S3 → 텍스트 수집
    # ──────────────────────────────
    def load_texts(self) -> None:
        """
        ① S3 버킷에서 prefix 아래 모든 키를 가져오고
        ② 확장자에 따라 JSON vs TXT 파싱 함수를 호출
        ③ self.texts 에 누적 저장
        """
        keys = self._s3.list_files_in_folder(self.prefix)
        if not keys:
            raise RuntimeError(f"❌ S3에 '{self.prefix}' 아래 파일이 없습니다.")

        # 파일 하나씩 순회
        for key in keys:
            if key.lower().endswith(".json"):
                # JSON → 장소 정보 추출
                self.texts.extend(self._s3.get_all_places_from_json(key))
            else:
                # TXT → 장소 정보 추출
                self.texts.extend(self._s3.get_all_places_from_txt(key))

        if not self.texts:
            raise RuntimeError("❌ 불러온 텍스트가 없습니다.")
        print(f"[Load] {len(self.texts):,}개의 문장 로드 완료")

    # ──────────────────────────────
    # 1-3. KoSimCSE 임베딩
    # ──────────────────────────────
    def embed(self) -> None:
        """
        ① self.texts 를 batch_size 단위로 잘라
        ② HuggingFaceEmbeddings.embed_documents() 호출
        ③ numpy array 로 모아서 L2 정규화
        """
        if not self.texts:
            raise RuntimeError("load_texts() 먼저 호출하세요.")

        batch_vectors: List[List[float]] = []           # 배치별 벡터 임시 저장
        for i in range(0, len(self.texts), self.batch_size):
            batch = self.texts[i : i + self.batch_size] # 슬라이싱
            vecs  = self._embedder.embed_documents(batch)
            batch_vectors.extend(vecs)
            # 진행률 출력 (ex: 128/240)
            print(f"[Embed] {min(i+self.batch_size, len(self.texts)):,}/{len(self.texts):,}")

        # python list → np.ndarray & L2 normalize
        self.vectors = normalize(np.asarray(batch_vectors, dtype="float32"), norm="l2")
        print("[Embed] 임베딩 + L2 정규화 완료")

    # ──────────────────────────────
    # 1-4. 결과 JSON 저장 (선택)
    # ──────────────────────────────
    def save_json(self) -> Path:
        """
        texts와 vectors를 1:1 매핑해 JSON(list[dict]) 으로 저장한 뒤
        파일 경로를 리턴
        """
        if len(self.vectors) == 0:
            raise RuntimeError("embed() 먼저 호출하세요.")

        # 부모 디렉터리 생성(없으면)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 리스트 컴프리헨션으로 [{"text": "...", "embedding": [...]}] 구조 생성
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"text": t, "embedding": v.tolist()}
                    for t, v in zip(self.texts, self.vectors)
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Save] JSON 저장 → {self.output_path}")
        return self.output_path

    # ──────────────────────────────
    # 1-5. 파이프라인(one-shot) 실행
    # ──────────────────────────────
    def run(self, save_json: bool=False) -> Tuple[List[str], np.ndarray]:
        """
        전체 파이프라인 수행:
          load_texts → embed → (선택) save_json
        리턴값은 (texts, vectors)
        """
        self.load_texts()
        self.embed()
        if save_json:
            self.save_json()
        return self.texts, self.vectors

# ────────────────────────────────────────────────────────────────────────────────
# 2. CLI 진입점
#    • `python embedding/embed_from_s3.py` 로 직접 실행하면 작동
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ▶️ 인스턴스 생성
    embedder = EmbedFromS3(
        prefix="raw_data/",                             # S3 key 접두사
        model_name="BM-K/KoSimCSE-roberta",             # KoSimCSE 모델
        batch_size=64,
        output_path=BASE_DIR / "data/embedding_data/embeddings.json",
    )

    # ▶️ 파이프라인 실행 (save_json=True → 결과 파일까지 저장)
    texts, vectors = embedder.run(save_json=True)

    # ▶️ 최종 요약 출력
    print("="*60)
    print(f"✅ 임베딩 완료! • 문장 수: {len(texts):,} • 벡터 shape: {vectors.shape}")
    print("="*60)
