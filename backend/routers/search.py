import logging

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query

from config import DEFAULT_TOP_K
from services.search_service import SearchService

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Search in indexed PDF files")
async def search_pdfs(
        query: str = Query(..., description="검색어"),
        top_k: int = Query(DEFAULT_TOP_K, description="반환할 결과 개수", ge=1, le=50)
):
    """
    인덱싱된 PDF 파일에서 검색

    - **query**: 검색어
    - **top_k**: 반환할 결과 개수 (기본값: 5, 최대: 50)

    인덱싱된 모든 PDF 파일에서 검색어와 의미적으로 유사한 내용을 찾습니다.
    """
    search_service = SearchService()

    # 인덱스 상태 확인
    status = search_service.get_status()
    if status["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail="검색 인덱스가 준비되지 않았습니다. 먼저 PDF 파일을 업로드하세요."
        )

    # 검색 실행
    results = search_service.search(query, top_k)

    if not results:
        return {"query": query, "results": [], "message": "검색 결과가 없습니다."}

    return {"query": query, "results": results}


@router.post("/file/", summary="Search in a specific PDF file without indexing")
async def search_single_pdf(
        file: UploadFile = File(...),
        query: str = Form(...),
        top_k: int = Form(DEFAULT_TOP_K, ge=1, le=50)
):
    """
    특정 PDF 파일에서만 검색 (인덱싱하지 않음)

    - **file**: 검색할 PDF 파일
    - **query**: 검색어
    - **top_k**: 반환할 결과 개수 (기본값: 5, 최대: 50)

    업로드된 단일 PDF 파일에서 검색어와 의미적으로 유사한 내용을 찾습니다.
    파일은 인덱스에 추가되지 않으며 임시로만 사용됩니다.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 검색할 수 있습니다.")

    # 파일 내용 읽기
    file_content = await file.read()

    search_service = SearchService()

    # 검색 실행
    results = search_service.search_file(file_content, file.filename, query, top_k)

    if not results:
        return {
            "query": query,
            "filename": file.filename,
            "results": [],
            "message": "검색 결과가 없습니다."
        }

    return {
        "query": query,
        "filename": file.filename,
        "results": results
    }


@router.get("/status/", summary="Get search service status")
async def get_status():
    """
    검색 서비스 상태 정보 조회

    인덱싱된 문서 수, 청크 수 등 검색 서비스의 현재 상태 정보를 반환합니다.
    """
    search_service = SearchService()
    return search_service.get_status()

