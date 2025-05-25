from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 불러오기
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 새로운 장소 정보 JSON
place_json = {
    "formatted_address": "대한민국 제주특별자치도 서귀포시 대정읍 가파리",
    "geometry": {
        "location": {
            "lat": 33.11972219999999,
            "lng": 126.2672222
        }
    },
    "name": "마라원짜장",
    "opening_hours": {
        "weekday_text": [
            "월요일: 오전 8:00 ~ 오후 6:00",
            "화요일: 오전 8:00 ~ 오후 6:00",
            "수요일: 오전 8:00 ~ 오후 6:00",
            "목요일: 오전 8:00 ~ 오후 6:00",
            "금요일: 오전 8:00 ~ 오후 6:00",
            "토요일: 오전 8:00 ~ 오후 6:00",
            "일요일: 오전 8:00 ~ 오후 6:00"
        ]
    },
    "price_level": 2,
    "rating": 3.3,
    "reviews": [
        {
            "rating": 5,
            "text": "제주 톳이 들어간 해물 짜장면이 특징적이었고 맛이 풍부했다."
        },
        {
            "rating": 3,
            "text": "고명으로 톳이 올라간 무난한 짜장면 맛."
        },
        {
            "rating": 4,
            "text": "경치는 매우 좋았고, 자장면과 짬뽕은 무난했다."
        },
        {
            "rating": 4,
            "text": "기대 없이 갔는데 맛이 좋아서 놀랐다."
        },
        {
            "rating": 3,
            "text": "위생은 아쉬웠지만 경치는 최고였다."
        }
    ],
    "types": [
        "restaurant",
        "food",
        "point_of_interest",
        "establishment"
    ]
}

# 리뷰 요약 만들기 (간단한 방식)
def summarize_reviews(reviews):
    lines = [r["text"] for r in reviews[:3]]
    return " ".join(lines)

summary = summarize_reviews(place_json["reviews"])
category = "음식점" if "restaurant" in place_json["types"] else "기타"

# 프롬프트 생성
prompt = f"""
다음은 여행지 정보입니다. 아래 6가지 여행 스타일에 대해 해당 장소가 적합(prefer)한지 또는 부적합(nonPrefer)한지를 판단하고, 각각 이유를 한 문장으로 설명해주세요.

여행 스타일: 힐링, 관광, 액티비티, 맛집 탐방, 가족 여행, 쇼핑

장소 정보:
- 이름: {place_json['name']}
- 위치: {place_json['formatted_address']}
- 분류: {category}
- 평점: {place_json['rating']}
- 설명: {summary}

출력 형식 예시:
힐링: prefer – 조용한 분위기와 자연 풍경이 마음을 안정시켜 줍니다.  
관광: prefer – 지역적 특성과 풍경이 관광 목적에 적합합니다.  
액티비티: nonPrefer – 활동적인 체험 요소가 부족합니다.  
맛집 탐방: prefer – 지역 특색 음식이 유명합니다.  
가족 여행: nonPrefer – 배를 타야 하는 접근성 때문에 불편할 수 있습니다.  
쇼핑: nonPrefer – 쇼핑 시설이 없습니다.

지금 장소에 대해 위 형식대로 결과를 작성해 주세요.
"""

# 추론
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 출력 정제
print("\n=== TinyLlama 출력 결과 ===\n")
lines = output_text.splitlines()
for line in lines:
    if any(style in line for style in ["힐링", "관광", "액티비티", "맛집", "가족", "쇼핑"]):
        print(line.strip())
