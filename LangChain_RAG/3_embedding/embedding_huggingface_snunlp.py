import os
import json
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings

def load_sentences(preprocessed_dir: str) -> list[str]:
    """
    지정된 폴더 내 모든 전처리된 텍스트 파일을 읽어
    메타데이터(격자점 정보)를 제외한 문장 리스트를 반환합니다.
    """
    sentences: list[str] = []
    pre_dir = Path(preprocessed_dir)

    for file_path in pre_dir.glob("preprocessed_data_grid_0012.txt"):
        content = file_path.read_text(encoding="utf-8")
        parts = content.split("\n\n")[1:]
        for part in parts:
            line = part.strip()
            if line:
                sentences.append(line)
    return sentences


def main():
    # 전처리된 텍스트가 저장된 디렉토리
    PREPROCESSED_DIR = Path(__file__).parent / "data" / "preprocessed_raw_data"

    # 문장 로드
    texts = load_sentences(str(PREPROCESSED_DIR))
    if not texts:
        print("전처리된 문장이 없습니다. 경로와 파일명을 확인하세요.")
        return

    # Hugging Face 임베딩 객체 생성 (로컬 모델)
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    # 문장 리스트를 배치로 임베딩 생성
    print(f"총 {len(texts)}개 문장에 대해 임베딩을 생성합니다...")
    vectors = embeddings.embed_documents(texts)

    # 결과를 저장할 디렉토리 생성
    output_dir = Path(__file__).parent / "data" / "embedding_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 텍스트와 임베딩을 로컬 JSON 파일로 저장
    output_path = output_dir / "embeddings.json"
    data = [{"text": txt, "embedding": vec} for txt, vec in zip(texts, vectors)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"임베딩 정보를 '{output_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
