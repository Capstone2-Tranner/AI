# summarize.py
# 한국어 리뷰 텍스트에 특화된 경량 KoBART 모델로 일괄 요약

from typing import List, Dict, Any
import torch
from transformers import pipeline, Pipeline

# GPU가 있으면 GPU 사용, 없으면 CPU 사용
_device = 0 if torch.cuda.is_available() else -1

# 1) 한국어 요약에 특화된 KoBART 모델 사용
#    파라미터 약 120M 수준으로 경량화됨
summarizer: Pipeline = pipeline(
    "summarization",
    model="ainize/kobart-summary",  # 한국어 요약 특화 모델
    tokenizer="ainize/kobart-summary",
    device=_device
)


def summarize_record(
    record: Dict[str, Any],
    max_length: int = 128,
    min_length: int = 30
) -> Dict[str, Any]:
    """
    단일 장소 레코드의 'review' 필드를 한국어 요약 모델로 요약하고 덮어씁니다.
    """
    text: str = record.get("review", "").strip()
    if not text:
        return record

    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    record["review"] = summary[0]["summary_text"].strip()
    return record


def summarize_records(
    records: List[Dict[str, Any]],
    max_length: int = 128,
    min_length: int = 30
) -> List[Dict[str, Any]]:
    """
    여러 장소 레코드 리스트에 대해 한국어 리뷰 요약을 일괄 적용하고 결과 리스트를 반환합니다.
    """
    for i, rec in enumerate(records):
        records[i] = summarize_record(rec, max_length, min_length)
    return records
