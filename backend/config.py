import os
import torch
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")

# 임베딩 모델 설정: 로컬 스냅샷이 있으면 사용하고, 없으면 HF Hub ID로 폴백 (HF 캐시 활용)
_LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "model", "ko-sroberta-multitask")
EMBEDDING_MODEL_NAME = (
    _LOCAL_MODEL_DIR
    if os.path.exists(os.path.join(_LOCAL_MODEL_DIR, "config.json"))
    else "jhgan/ko-sroberta-multitask"
)
MAX_LENGTH = 512

# CPU 성능 최적화 설정 (코어 수에 맞게 자동 조정, 최대 4)
torch.set_num_threads(min(os.cpu_count() or 1, 4))

# 검색 설정
DEFAULT_TOP_K = 5

# 배치 처리 설정
EMBEDDING_BATCH_SIZE = 8  # CPU 메모리에 맞게 조정

# FAISS 인덱스 설정
USE_GPU_FOR_FAISS = False  # CPU만 사용

# 업로드 제한
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 단일 파일 최대 100MB

# 운영 SPA 호스팅 경로 (frontend/dist). 존재할 때만 main.py가 마운트.
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    os.path.normpath(os.path.join(BASE_DIR, os.pardir, "frontend", "dist")),
)

# 인덱스 파일 경로 (청크는 특수문자 안전한 parquet로 저장)
INDEX_FILE_PATH = os.path.join(INDEX_DIR, "pdf_index.faiss")
CHUNKS_FILE_PATH = os.path.join(INDEX_DIR, "pdf_chunks.parquet")
METADATA_FILE_PATH = os.path.join(INDEX_DIR, "pdf_metadata.parquet")


# 디렉토리 생성 함수
def create_directories():
    """애플리케이션에 필요한 디렉토리 생성"""
    directories = [STATIC_DIR, DATA_DIR, INDEX_DIR, PDF_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)