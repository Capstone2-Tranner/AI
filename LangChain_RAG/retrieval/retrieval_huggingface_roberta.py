# 이 파일의 목적:
#   1. 사용자가 질문을 하면 그 질문과 가장 비슷한 장소 정보를 찾아주는 기능
#   2. 얼마나 비슷한지 점수도 함께 알려줌 (1에 가까울수록 더 비슷함)
#
# 어떻게 동작하나요?
#   1. 준비 단계:
#      - 이전에 저장해둔 장소 정보들을 불러옴
#      - 파일이 없으면 새로 만듦
#   
#   2. 검색 단계:
#      - 사용자 질문을 컴퓨터가 이해할 수 있는 숫자(벡터)로 바꿈
#      - 저장된 장소 정보들 중에서 가장 비슷한 것 k개를 찾음
#      - 얼마나 비슷한지 점수를 계산해서 알려줌
#
# 필요한 파일들:
#   - faiss_index_hnsw.idx : 빠른 검색을 위해 만든 특별한 파일
#   - texts.json : 실제 장소 정보가 담긴 파일
#   - store_vector_huggingface_roberta.py : 위 파일들을 만드는 프로그램
#
# 사용 예시:
#   $ python retrieval_huggingface_roberta.py "제주도 짜장면 맛집 추천"
#   → 가장 비슷한 5개의 장소 정보를 보여줌

from __future__ import annotations  # 파이썬 타입 힌트를 더 유연하게 쓸 수 있게 해줌

# 파이썬 기본 도구들
import json
from pathlib import Path
from typing import List, Tuple

# 외부 라이브러리들
import faiss                                   # 빠른 검색을 위한 페이스북 AI 도구
import numpy as np
from sklearn.preprocessing import normalize    # 벡터 계산을 위한 도구
from langchain_community.embeddings import HuggingFaceEmbeddings  # 한국어 처리 모델

# 내부 파일 불러오기 설정
import sys
BASE_DIR      = Path(__file__).resolve().parents[1]          # 프로젝트 메인 폴더
VECTOR_DIR    = BASE_DIR / "4_store_vector"                  # 벡터 저장 폴더
sys.path.append(str(VECTOR_DIR))                             # 파일 찾을 수 있게 경로 추가

from store_vector_huggingface_roberta import VectorStoreBuilder  # 벡터 저장소 만드는 도구


class VectorStoreRetriever:
    """
    사용자 질문을 받아서 비슷한 장소 정보를 찾아주는 클래스
    - KoSimCSE-Roberta: 한국어를 이해하는 AI 모델
    - FAISS-HNSW: 빠르게 비슷한 정보를 찾는 도구
    """

    def __init__(
        self,
        db_dir: Path | str = "data/vector_db",        # 데이터 저장 폴더
        model_name: str = "BM-K/KoSimCSE-roberta",    # 한국어 처리 모델
    ):
        # 1. 파일 경로 설정
        self.db_dir   = Path(db_dir)
        self.idx_path = self.db_dir / "faiss_index_hnsw.idx"  # 검색 인덱스 파일
        self.txt_path = self.db_dir / "texts.json"            # 실제 장소 정보 파일

        # 2. 나중에 쓸 변수들 준비
        self.index: faiss.Index | None = None  # 검색 도구
        self.texts: List[str]          = []    # 장소 정보들

        # 3. 한국어 처리 모델 준비
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def load(self) -> None:
        """
        필요한 파일들을 불러옴
        - 파일이 없으면 새로 만듦
        """
        if self.idx_path.exists() and self.txt_path.exists():
            # 파일이 있으면 바로 불러옴
            self._load_files()
        else:
            # 파일이 없으면 새로 만들고 불러옴
            print("[알림] 필요한 파일이 없어서 새로 만듭니다...")
            VectorStoreBuilder(db_dir=self.db_dir).run(batch_size=64)
            self._load_files()

    def _load_files(self) -> None:
        """
        두 가지 파일을 불러옴:
        1. 검색 인덱스 파일
        2. 실제 장소 정보 파일
        """
        self.index = faiss.read_index(str(self.idx_path))
        with self.txt_path.open("r", encoding="utf-8") as f:
            self.texts = json.load(f)
        print(f"[알림] {len(self.texts):,}개의 장소 정보를 불러왔습니다")

    def query(
        self,
        sentence: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        사용자 질문과 비슷한 장소 정보를 찾아줌

        입력:
            sentence : 사용자 질문 (예: "제주도 맛집 추천해줘")
            k : 몇 개의 결과를 보여줄지 (기본값 5개)

        출력:
            [(장소정보1, 유사도1), (장소정보2, 유사도2), ...] 형태의 리스트
            - 유사도는 1에 가까울수록 더 비슷함
            - -1에 가까울수록 정반대
        """
        # 1. 파일을 안 불러왔으면 에러
        if self.index is None or not self.texts:
            raise RuntimeError("load() 함수를 먼저 실행해주세요!")

        # 2. 사용자 질문을 컴퓨터가 이해할 수 있는 숫자로 바꿈
        vec = self._embedder.embed_query(sentence)                
        vec = normalize(np.asarray([vec], dtype="float32"), norm="l2")  

        # 3. 가장 비슷한 장소 정보 k개를 찾음
        distances, indices = self.index.search(vec, k)            

        # 4. 거리를 유사도 점수로 바꿈 (1에 가까울수록 비슷)
        similarities = 1 - distances[0] / 2                              

        # 5. 결과를 보기 좋게 만들어서 반환
        return [
            (self.texts[i], float(sim))
            for i, sim in zip(indices[0], similarities)
        ]


# 프로그램 테스트용 코드
if __name__ == "__main__":  
    import sys

    # 검색할 내용 준비
    query_sentence = (
        " ".join(sys.argv[1:]) or
        "짜장면을 먹고싶은데, 위치는 제주특별자치도이고 restaurant이고 영업 시간은 평일 주말 모두 하는 곳 알려줘"
    )

    # 검색 실행
    retriever = VectorStoreRetriever()
    retriever.load()                     # 파일 불러오기
    results = retriever.query(query_sentence, k=5)

    # 결과 예쁘게 출력
    print(f"\n📌 검색어: {query_sentence}\n")
    for rank, (text, score) in enumerate(results, 1):
        print(f"{rank:>2}. (유사도: {score:.3f}) {text}")
