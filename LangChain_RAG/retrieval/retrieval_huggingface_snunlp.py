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
import json
from pathlib import Path
import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_texts(texts_file: Path) -> list[str]:
    with texts_file.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    base_dir = Path(__file__).parent / 'data' / 'vector_db'
    index_path = base_dir / 'faiss_index.idx'
    texts_path = base_dir / 'texts.json'

    # 1) FAISS 인덱스 로드 (Windows 한글 경로 이슈 회피)
    # 파일을 Python I/O로 읽은 뒤 numpy 배열로 변환, deserialize
    raw = index_path.read_bytes()
    buf = np.frombuffer(raw, dtype='uint8')
    index = faiss.deserialize_index(buf)
    print(f'인덱스 로드 완료. 총 벡터 수: {index.ntotal}')

    # 2) 텍스트 매핑 로드
    texts = load_texts(texts_path)
    print(f'텍스트 매핑 로드 완료. 총 문장 수: {len(texts)}')

    # 3) 임베딩 객체 생성 (import 경로 업데이트)
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    # 4) 사용자 입력 쿼리
    query = input('검색할 문장을 입력하세요: ')
    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector], dtype='float32')

    # 5) 유사도 검색
    faiss.normalize_L2(query_vector)
    k = 5
    D, I = index.search(query_vector, k)

    # 6) 결과 출력
    print(f'=== Top {k} 검색 결과 ===')
    for dist, idx in zip(D[0], I[0]):
        print(f'거리: {dist:.4f}, 문장: {texts[idx]}')

if __name__ == '__main__':
    main()

# 검색 결과로 짜장면 집 검색 -> 짜장면집이 높은 유사도로 검색되는 것이 아니라(리뷰에 짜장이라는 키워드가 더 많이 들어간 곳으로 검색됨)

# 저장 방식을 L2가 아닌 클러스터링 방식으로 카테고리별로 묶어서 검색해야 하나?
