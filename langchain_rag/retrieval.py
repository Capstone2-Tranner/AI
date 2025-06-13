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
from pathlib import Path
import json
from typing import List, Union, Dict, Any
import os
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings



class VectorStoreRetriever:
    def __init__(self,
                 db_dir: Union[Path, str] = None,
                 model_name: str = "BM-K/KoSimCSE-roberta"):
        # (디버그) 현재 작업 디렉터리(CWD) 확인
        print("DEBUG: 현재 CWD =", os.getcwd())

        # 이 파일(retrieval.py)의 절대경로 → …/Capstone2/langchain_rag/retrieval.py
        this_file = Path(__file__).resolve()
        langchain_rag_dir = this_file.parent  # → …/Capstone2/langchain_rag

        self.db_dir = langchain_rag_dir / "utils" / "data" / "embedding_data"


        # 실 파일이 있는 경로로 idx_path를 설정
        self.idx_path = self.db_dir / "faiss_index.index"
        self.txt_path = self.db_dir / "faiss_index.meta.json"

        self.index: faiss.Index | None = None
        self.records: List[Dict[str, Any]] = []
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def _load_index(self) -> None:
        print("DEBUG: self.db_dir =", self.db_dir)
        print("DEBUG: self.idx_path.resolve() =", self.idx_path.resolve())

        if self.index is None:
            print("로드하려는 FAISS 인덱스 경로:", self.idx_path)
            if not self.idx_path.exists():
                print(f"DEBUG: 인덱스 파일 존재 여부 → {self.idx_path.resolve()} : {self.idx_path.exists()}")
                raise FileNotFoundError(f"인덱스 파일이 없습니다: {self.idx_path}")
            self.index = faiss.read_index(str(self.idx_path))

    def _load_records(self) -> None:
            """
            JSON 파일에서 "texts" 키에 해당하는 리스트를 self.records에 저장합니다.
            """
            if self.records:
                return

            if not self.txt_path.exists():
                raise FileNotFoundError(f"메타데이터 JSON 파일이 없습니다: {self.txt_path}")

            with open(self.txt_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 최상위가 딕셔너리여야 하고, "texts" 키가 있어야 함
            if not isinstance(data, dict) or "texts" not in data:
                raise ValueError(f"메타데이터 JSON 형식이 잘못되었습니다: 'texts' 키를 찾을 수 없음")

            texts_list = data["texts"]
            if not isinstance(texts_list, list):
                raise ValueError(f"'texts' 필드가 리스트가 아닙니다: {type(texts_list)}")

            # 각 요소가 문자열인지 검증(선택 사항)
            for i, item in enumerate(texts_list):
                if not isinstance(item, str):
                    raise ValueError(f"'texts' 리스트의 {i}번째 항목이 문자열이 아닙니다: {type(item)}")

            self.records = texts_list

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        주어진 쿼리 문장을 임베딩하여 FAISS에서 유사도가 높은 상위 top_k개의 장소 정보를 반환합니다.

        Args:
            query: 검색용 자연어 쿼리 문자열
            top_k: 상위 몇 개 결과를 반환할지
        Returns:
            List of dicts, each containing keys:
            - place_id (str)
            - region (str)
            - category (str)
            - location: {lat: float, lng: float}
            - rating: float
            - price_level: (optional) int or null
            - opening_hours: { weekday_text: List[str] }
            - review: str
        """
        # 인덱스 및 레코드 로드
        self._load_index()
        self._load_records()

        # 쿼리 임베딩 획득 및 검색
        q_emb = self._embedder.embed_query(query)
        xq = np.array(q_emb, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(xq, top_k)

        # 인덱스를 레코드로 매핑하여 반환
        results: List[Dict[str, Any]] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.records):
                results.append(self.records[idx])
        return results
