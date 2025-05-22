'''
5.
    목적:
        vector store에서 유사한 벡터를 검색하고 유사도를 계산한다.
    방법:
        vector store에서 쿼리와 유사한 벡터를 검색하고, 유사도 점수를 계산한다.
        검색 결과는 유사도에 따라 정렬된다.
    후속 처리:
        검색된 결과는 make_prompt에서 프롬프트 생성에 사용된다.
'''
#!/usr/bin/env python3
# =============================================================================
#  LangChain_RAG / 4_store_vector / vector_db / retriever.py
#  ---------------------------------------------------------------------------
#  ■ 목적(Purpose)
#     - 저장해 둔 벡터 DB(FAISS + HNSW)에서 질문(Query)과 **가장 유사**한
#       문장(장소 정보 텍스트)을 k개 반환한다.
#     - 반환 시 코사인 유사도(cosine-similarity) 점수도 함께 넘겨줘,
#       downstream 단계(make_prompt 등)에서 중요도 순으로 활용할 수 있게 함.
#
#  ■ 동작 개요
#     1. load()   : ➊ 인덱스(.idx)와 ➋ 원본 문장 매핑(texts.json)을 메모리로 로드
#                   ‣ 둘 중 하나라도 없으면 VectorStoreBuilder 를 호출해 새로 생성
#     2. query()  : ➊ 입력 문장 → KoSimCSE 임베딩 → L2 정규화
#                   ➋ FAISS index.search() 로 top-k L2 거리 검색
#                   ➌ 거리(L2) → 코사인 유사도 로 변환 후 (문장, 점수) 리스트 반환
#
#  ■ 의존 파일
#     - ../data/vector_db/faiss_index_hnsw.idx   : FAISS HNSW 인덱스
#     - ../data/vector_db/texts.json             : 인덱스와 동일 순서의 문장 배열
#     - 4_store_vector/store_vector_huggingface_roberta.py : 인덱스 빌더
#
#  ■ CLI 예시
#     $ python -m vector_db.retriever "제주도 짜장면 맛집 추천"
#     → 상위 5개 문장과 유사도가 터미널에 출력
# =============================================================================

from __future__ import annotations  # PEP-563: postponed evaluation of annotations

# ── 표준 라이브러리 ───────────────────────────────────────────────────────────
import json
from pathlib import Path
from typing import List, Tuple

# ── 서드파티 라이브러리 ────────────────────────────────────────────────────────
import faiss                                   # Facebook AI Similarity Search
import numpy as np
from sklearn.preprocessing import normalize    # 벡터 L2 정규화
from langchain_community.embeddings import HuggingFaceEmbeddings  # KoSimCSE

# ── 내부 모듈 경로 세팅 ────────────────────────────────────────────────────────
import sys                          # import 우회
from Capstone2.langchain_rag.utils.store_vector import VectorStoreBuilder  # 인덱스 생성기


# =============================================================================
#  VectorStoreRetriever 클래스 정의
# =============================================================================
class VectorStoreRetriever:
    """
    KoSimCSE-Roberta 임베더 + FAISS-HNSW 인덱스를 이용해
    • 질문(자연어) → top-k 유사 문장 & 코사인 유사도 반환
    """

    # --------------------------------------------------------------------- #
    # 1) 생성자: 경로/모델 설정 & 임베더 초기화
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        db_dir: Path | str = "data/vector_db",        # 인덱스/텍스트 저장 폴더
        model_name: str = "BM-K/KoSimCSE-roberta",    # 임베딩 모델
    ):
        # (1) 파일 경로 객체 세팅
        self.db_dir   = Path(db_dir)
        self.idx_path = self.db_dir / "faiss_index_hnsw.idx"  # 인덱스 바이너리
        self.txt_path = self.db_dir / "texts.json"            # 문장 매핑

        # (2) 지연 로드(lazy-load)용 캐시 변수
        self.index: faiss.Index | None = None  # FAISS 인덱스 객체
        self.texts: List[str]          = []    # 인덱스와 동일 순서의 문장 리스트

        # (3) 질의(Query)용 KoSimCSE 임베더
        #     embed_query() / embed_documents() 두 메서드를 제공
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    # --------------------------------------------------------------------- #
    # 2) 인덱스 + 텍스트 매핑 로드 (없으면 자동 생성)
    # --------------------------------------------------------------------- #
    def load(self) -> None:
        """
        인덱스(.idx)와 문장 매핑(texts.json)을 메모리로 로드한다.
        두 파일 중 하나라도 없으면 VectorStoreBuilder.run() 으로 새로 만든다.
        """
        if self.idx_path.exists() and self.txt_path.exists():
            # 2-A. 이미 빌드돼 있으면 바로 로드
            self._load_files()
        else:
            # 2-B. 없으면 빌드 후 로드
            print("[Retrieval] 인덱스 파일이 없어 새로 만듭니다…")
            VectorStoreBuilder(db_dir=self.db_dir).run(batch_size=64)
            self._load_files()

    def _load_files(self) -> None:
        """
        • faiss.read_index() : 인덱스 바이너리 → faiss.Index 객체
        • texts.json         : list[str] 로 로드
        """
        self.index = faiss.read_index(str(self.idx_path))
        with self.txt_path.open("r", encoding="utf-8") as f:
            self.texts = json.load(f)
        print(f"[Retriever] 인덱스 & 텍스트 {len(self.texts):,}개 로드 완료")

    # --------------------------------------------------------------------- #
    # 3) 질의(Query) 수행 → 상위 k개 (문장, 유사도) 반환
    # --------------------------------------------------------------------- #
    def query(
        self,
        sentence: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Parameters
        ----------
        sentence : str
            사용자의 검색 질문(자연어)
        k : int
            반환할 문장 개수 (기본 5)

        Returns
        -------
        List[Tuple[str, float]]
            각 원소는 (문장, cosine-similarity) 형태
            • 유사도는 1.0(완전 동일) ~ -1.0(정반대)에 가까울수록 유사함
        """
        # 3-A. 인덱스가 메모리에 없는 경우 예외 처리
        if self.index is None or not self.texts:
            raise RuntimeError("load() 를 먼저 호출하세요.")

        # 3-B. 쿼리 문장 → KoSimCSE 벡터 변환
        vec = self._embedder.embed_query(sentence)                # 1차원 list[float]
        vec = normalize(np.asarray([vec], dtype="float32"), norm="l2")  # (1, D)

        # 3-C. FAISS 검색 (metric=L2, index type=HNSWFlat)
        #      search() 는 '거리'가 작을수록 유사한 인덱스를 반환
        distances, indices = self.index.search(vec, k)            # (1,k) 행렬

        # 3-D. L2 거리 → 코사인 유사도로 변환
        #      ‖a-b‖² = 2(1 - cosθ)  ⇒  cosθ = 1 - dist/2
        dists = 1 - distances[0] / 2                              # shape (k,)

        # 3-E. (문장, 유사도) 튜플 리스트로 가공 후 반환
        return [
            (self.texts[i], float(sim))
            for i, sim in zip(indices[0], dists)
        ]


# =============================================================================
# 4. CLI 테스트 (python -m vector_db.retriever "검색어")
# =============================================================================
if __name__ == "__main__":  # pragma: no cover (테스트 커버리지 제외)
    import sys

    # ▶︎ 터미널 인자 합쳐서 query 문자열 생성
    query_sentence = (
        " ".join(sys.argv[1:]) or
        "짜장면을 먹고싶은데, 위치는 제주특별자치도이고 restaurant이고 영업 시간은 평일 주말 모두 하는 곳 알려줘"
    )

    # ▶︎ 로더 & 쿼리
    retriever = VectorStoreRetriever()
    retriever.load()                     # 인덱스/문장 로드
    results = retriever.query(query_sentence, k=5)

    # ▶︎ 결과 pretty-print
    print(f"\n📌 Query: {query_sentence}\n")
    for rank, (text, score) in enumerate(results, 1):
        print(f"{rank:>2}. ({score:.3f}) {text}")

"""
─────────────────────────────────────────────────────────────────────────────
※ 참고 ①  코사인 vs L2 거리
    - index.search()가 반환하는 'distances'는 L2 제곱거리(‖a-b‖²).
    - HNSWFlat(L2) = 2(1-cosθ) 공식 이용해 코사인 유사도로 변환.

※ 참고 ②  결과가 리뷰 문장 위주로 뜨는 현상
    - 현재 raw JSON→텍스트 추출 시 리뷰가 포함되어 있어 가게 메타정보보다
      리뷰 키워드가 유사도에 더 크게 작용할 수 있음.
    - 해결책: ❶ 원본 텍스트 구조(메타정보 우선) 재가공,
             ❷ 사용자 쿼리를 “평점·운영시간·카테고리 강조” 프레이즈로 확장.
─────────────────────────────────────────────────────────────────────────────
"""


# 검색 결과로 짜장면 집 검색 -> 짜장면집이 높은 유사도로 검색되는 것이 아니라(리뷰에 짜장이라는 키워드가 더 많이 들어간 곳으로 검색됨)
#해결 방법 :  사용자가 "마라도 맛집 알려줘"라고 검색했을 때,
#단순히 "마라도 맛집"만 임베딩하지 말고, 다음처럼 확장해서 **"가게 정보 중심의 의미"**를 모델에게 학습시키는 것
# ex) "마라도에서 위치, 운영시간, 평점 등 가게 정보가 잘 정리된 맛집을 알고 싶어요."
#이렇게 하면 임베딩 모델이 문장 내의 **정보성 키워드 (운영시간, 위치, 평점 등)**에 더 집중하게 되어, 리뷰 위주가 아닌 메타 정보 중심 문장과의 유사도를 더 높게 평가할 수 있다
# raw데이터의 형태를 조금 바꿔봐야 할 것 같음 -> 쿼리를 아무리 잘 짜도 리뷰의 비중이 너무 높은듯듯


#1. 지역 리스트 랜덤 뽑기
#2. retrieval 클래스화
#3. retrieval 검색 문장 원하는 정보에 집중할 수 있게 생성하기(위치, 카테고리) -> 프론트에서 받는 정보를 통해 어떤 law 데이터를 retrieval해야 할지 생각해야함함
#4. 테스트는 local에서 뽑고, 실제는 s3에서 뽑아올 수 있게 하기

