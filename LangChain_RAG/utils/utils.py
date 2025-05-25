'''
	목적:
		여러 모듈에서 공통으로 사용되는 유틸리티 함수들
	방법:
		Data preprocessing 함수, logging 함수, 에러 핸들링 함수 등을 정의한다.
'''
import logging

def setup_logger(name: str = __name__, log_file: str = "data_collection.log", level: int = logging.INFO) -> logging.Logger:
    """
    이름과 파일 경로를 받아 로거를 설정하고 반환합니다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        # 파일 핸들러
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(level)
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
