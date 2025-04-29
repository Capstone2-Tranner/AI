'''
2.
    목적: 
        raw_data_load로 얻은 raw data를 적절히 분할한다.
    방법: 
        구역 단위(구단위 or 시단위)로 장소 데이터를 split 하거나, 300줄로 split 하거나 여러 경우의 수가 있다.
        이 부분에서 여러 경우로 테스트 해봐서 최적의 split 방법을 골라야한다.
    후속 처리: 
        split된 data(split data)를 embedding.py를 통해 vector로 embedding한다.
'''