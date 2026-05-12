from fastapi import APIRouter
from routers.upload import router as upload_router
from routers.search import router as search_router


# 모든 라우터를 하나로 합침
router = APIRouter()
router.include_router(upload_router)
router.include_router(search_router)