import os
import logging

from typing import List
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, HTTPException

from config import PDF_DIR
from services.pdf_service import PDFService
from services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
    responses={404: {"description": "Not found"}},
)


@router.post("/pdf/", summary="Upload PDF files for indexing")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    PDF 파일 업로드 및 인덱싱

    - **files**: 업로드할 PDF 파일 목록

    업로드된 PDF 파일을 저장하고 내용을 추출하여 검색 인덱스에 추가합니다.
    """
    pdf_service = PDFService()
    search_service = SearchService()

    saved_files = []
    skipped_files = []

    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                skipped_files.append(file.filename)
                continue

            # 파일 내용 읽기
            file_content = await file.read()

            # 파일 저장
            saved_path = pdf_service.save_uploaded_pdf(file_content, file.filename)
            saved_files.append(saved_path)

        if not saved_files:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No valid PDF files uploaded"}
            )

        # 저장된 PDF 파일 인덱싱
        success = search_service.index_pdf_files(saved_files)

        if not success:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to index PDF files"}
            )

        return {
            "status": "success",
            "message": f"{len(saved_files)} PDF files uploaded and indexed successfully",
            "indexed_files": [os.path.basename(f) for f in saved_files],
            "skipped_files": skipped_files
        }
    except Exception as e:
        logger.error(f"Error in upload_pdfs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

