import os
import torch
import logging
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# CPU 전용 모드 설정 - 가장 먼저 설정해야 함
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # 시스템 CPU 코어 수에 맞게 조정

from config import create_directories, STATIC_DIR
from routers import router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# 필요한 디렉토리 생성
create_directories()

# CPU 전용 설정 확인
logger.info(f"CUDA is available: {torch.cuda.is_available()}")
logger.info(f"Using CPU only mode with {torch.get_num_threads()} threads")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="PDF 시맨틱 검색 API (CPU 전용)",
    description="ko-sroberta-multitask 임베딩 모델과 FAISS를 사용한 한국어 PDF 시맨틱 검색 시스템 (CPU 최적화)",
    version="1.0.0"
)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)

# 정적 파일 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    웹 인터페이스 홈페이지 반환
    """
    html_file_path = os.path.join(STATIC_DIR, "index.html")

    # 정적 디렉토리에 index.html이 없는 경우 간단한 HTML 반환
    if not os.path.exists(html_file_path):
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>PDF 시맨틱 검색 (CPU 전용)</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    h1 { color: #2c3e50; }
                    a { color: #3498db; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <h1>PDF 시맨틱 검색 API (CPU 전용)</h1>
                <p>PDF 파일에서 의미 기반 검색을 수행할 수 있는 API 서비스입니다.</p>
                <p>이 서비스는 CPU만을 사용하여 최적화되었습니다.</p>
                <p><a href="/docs">API 문서 보기</a></p>
            </body>
        </html>
        """

    # index.html 파일이 있는 경우 해당 파일 내용 반환
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    전역 예외 처리 핸들러
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"내부 서버 오류: {str(exc)}"}
    )


if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

