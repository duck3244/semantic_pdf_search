import os
import torch
import logging
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# CPU 전용 모드 설정 - 가장 먼저 설정해야 함
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # 시스템 CPU 코어 수에 맞게 조정

from config import create_directories, STATIC_DIR, FRONTEND_DIST
from routers import router
from models.embedding import EmbeddingModel

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 임베딩 모델 cold start 사전 해소: 첫 사용자 요청에 ~10초 지연이 발생하지 않도록
    # startup 시점에 1회 추론을 돌려 모델 가중치를 메모리에 적재.
    try:
        EmbeddingModel().get_embedding("warmup")
        logger.info("Embedding model warmup complete")
    except Exception as e:
        logger.warning(f"Embedding model warmup failed: {e}")
    yield


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="PDF 시맨틱 검색 API (CPU 전용)",
    description="ko-sroberta-multitask 임베딩 모델과 FAISS를 사용한 한국어 PDF 시맨틱 검색 시스템 (CPU 최적화)",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS 설정: credentials를 사용하지 않으므로 와일드카드는 허용되지만,
# 운영 환경에서는 ALLOWED_ORIGINS 환경변수로 화이트리스트를 좁히는 것을 권장.
_allowed = os.environ.get("ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if _allowed == "*" else [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 라우터 등록 (정적/SPA 마운트보다 먼저 등록되어 API가 우선 매칭됨)
app.include_router(router)


# 운영 SPA 마운트: frontend/dist가 존재하면 그 안의 정적 파일을 / 에서 서빙.
# html=True 옵션은 디렉토리 GET 시 index.html을 반환 → Vue 진입점.
# (Vue Router는 hash 모드라 서버 측 라우팅 fallback이 필요 없음)
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
    logger.info(f"Serving SPA from {FRONTEND_DIST}")
else:
    logger.info(f"SPA dist not found at {FRONTEND_DIST}; API-only mode")

    @app.get("/", response_class=HTMLResponse)
    async def api_only_home():
        return (
            "<html><body style='font-family:sans-serif;max-width:720px;margin:2rem auto;'>"
            "<h1>PDF 시맨틱 검색 API</h1>"
            "<p>운영 SPA 빌드가 마운트되지 않았습니다. "
            "<code>cd frontend && npm run build</code> 후 서버를 재시작하세요.</p>"
            "<p><a href='/docs'>API 문서 보기</a></p>"
            "</body></html>"
        )


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
    # 운영 기본값은 reload=False (app.log 자기참조로 인한 무한 reload 루프 회피).
    # 개발 시에는 RELOAD=1 환경변수로 켜고, 로그/인덱스 파일은 watcher에서 제외.
    reload = os.environ.get("RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        reload_excludes=["*.log", "data/*", "data/**/*"] if reload else None,
    )

