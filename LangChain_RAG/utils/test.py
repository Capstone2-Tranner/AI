# classify.py
# GPT4All을 사용해 장소 JSON을 받아 여행 스타일 적합도(prefer/nonPrefer)와 간단한 이유를 출력합니다.

import json
import os
import sys
from typing import Dict, Any, List
from gpt4all import GPT4All

# 1) 모델 경로 설정 및 확인
DEFAULT_MODEL_PATH = "models/gpt4all-lora-quantized.bin"

# 모델 파일이 없으면 사용자에게 안내 후 종료
def ensure_model(path: str):
    if not os.path.isfile(path):
        sys.stderr.write(
            f"모델 파일을 찾을 수 없습니다: {path}\n"
            "아래 주소에서 'gpt4all-lora-quantized.bin' 파일을 다운로드하여 해당 경로에 놓으세요:\n"
            "https://gpt4all.io/models/ggml/gpt4all-lora-quantized.bin\n"
        )
        sys.exit(1)

# 2) GPT4All 모델 로드

def load_model(model_path: str) -> GPT4All:
    ensure_model(model_path)
    return GPT4All(
        "gpt4all-lora-quantized",  # 모델 식별자
        model_path=model_path,       # 모델 파일 경로
        allow_download=False         # 자동 다운로드 비활성화
    )

# 분류용 프롬프트 템플릿
PROMPT_TEMPLATE = '''다음은 장소 정보의 JSON입니다:
```json
{place_json}
```
아래 6가지 여행 스타일에 대해 이 장소가 적합(prefer)한 스타일과 부적합(nonPrefer)한 스타일을 구분하여, 이유를 간단히 포함한 한 문장으로 출력해주세요:
• 힐링
• 관광
• 액티비티
• 맛집 탐방
• 가족 여행
• 쇼핑
'''


def classify_styles(llm: GPT4All, record: Dict[str, Any]) -> Dict[str, str]:
    """
    단일 장소 레코드를 받아 6가지 여행 스타일의 prefer/nonPrefer 구분과 이유를 반환합니다.
    """
    place_json = json.dumps(record, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(place_json=place_json)

    response = llm.generate(
        prompt,
        max_tokens=512,
        temperature=0.0,
        streaming=False
    )
    text = response.strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result: Dict[str, str] = {}
    for line in lines:
        if ':' in line:
            key, rest = line.split(':', 1)
            result[key.strip()] = rest.strip()
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify travel styles using GPT4All"
    )
    parser.add_argument('--input',  default='vector_db/texts.json', help='원본 JSON 경로')
    parser.add_argument('--output', default='vector_db/styles.json', help='저장할 JSON 경로')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, help='GPT4All 모델 파일 경로')
    args = parser.parse_args()

    # 모델 로드
    llm = load_model(args.model_path)

    # 입력 JSON 로드
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            records: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(f"입력 파일을 찾을 수 없습니다: {args.input}\n")
        sys.exit(1)

    # 스타일 분류
    outputs: List[Dict[str, Any]] = []
    for rec in records:
        styles = classify_styles(llm, rec)
        rec['styles'] = styles
        outputs.append(rec)

    # 결과 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Classified styles for {len(outputs)} records and saved to {args.output}")
