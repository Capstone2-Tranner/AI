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

import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreRetriever:
    def __init__(
        self,
        db_dir: Union[Path, str] = "data/vector_db",
        model_name: str = "BM-K/KoSimCSE-roberta",
    ):
        # (1) 파일 경로 객체 세팅
        self.db_dir = Path(db_dir)
        self.idx_path = self.db_dir / "faiss_index_hnsw.idx"
        self.txt_path = self.db_dir / "texts.json"

        # (2) 지연 로드(lazy-load)용 캐시 변수
        self.index: faiss.Index | None = None
        self.records: List[Dict[str, Any]] = []  # texts.json에 저장된 장소 정보 리스트

        # (3) 질의(Query)용 KoSimCSE 임베더
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def _load_index(self) -> None:
        """
        FAISS 인덱스를 파일에서 읽어와 초기화합니다.
        """
        if self.index is None:
            self.index = faiss.read_index(str(self.idx_path))

    def _load_records(self) -> None:
        """
        색인된 문장뿐 아니라 장소 정보 딕셔너리 리스트를 JSON 파일에서 읽어옵니다.
        texts.json에는 각 문장별 메타정보(장소ID, 지역, 카테고리, 위치, 평점 등)를 담고 있어야 합니다.
        """
        if not self.records:
            with open(self.txt_path, encoding="utf-8") as f:
                self.records = json.load(f)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
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
