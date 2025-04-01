import os
import torch
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "model/ko-sroberta-multitask"
MAX_LENGTH = 512

# CPU 성능 최적화 설정
# 멀티스레딩 설정 (Intel CPU를 위한 최적화)
torch.set_num_threads(4)  # 시스템에 맞게 조정 (예: CPU 코어 수에 따라)

# 검색 설정
DEFAULT_TOP_K = 5

# 배치 처리 설정
EMBEDDING_BATCH_SIZE = 8  # CPU 메모리에 맞게 조정

# FAISS 인덱스 설정
USE_GPU_FOR_FAISS = False  # CPU만 사용

# 인덱스 파일 경로
INDEX_FILE_PATH = os.path.join(INDEX_DIR, "pdf_index.faiss")
CHUNKS_FILE_PATH = os.path.join(INDEX_DIR, "pdf_chunks.csv")
METADATA_FILE_PATH = os.path.join(INDEX_DIR, "pdf_metadata.csv")


# 디렉토리 생성 함수
def create_directories():
    """애플리케이션에 필요한 디렉토리 생성"""
    directories = [STATIC_DIR, DATA_DIR, INDEX_DIR, PDF_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)