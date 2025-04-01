import faiss
import logging
import numpy as np
import pandas as pd

from typing import List, Dict, Any, Tuple, Optional

from config import PDF_DIR, DEFAULT_TOP_K
from models.faiss_index import FAISSIndex
from services.pdf_service import PDFService
from models.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class SearchService:
    """
    PDF 검색 기능을 제공하는 서비스
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.faiss_index = FAISSIndex()
        self.pdf_service = PDFService()


    def index_pdf_files(self, pdf_files: List[str]) -> bool:
        """
        PDF 파일을 인덱싱

        Args:
            pdf_files (List[str]): 인덱싱할 PDF 파일 경로 리스트

        Returns:
            bool: 인덱싱 성공 여부
        """
        try:
            # PDF 처리 및 임베딩 생성
            chunks_df, embeddings, metadata = self.pdf_service.process_pdf_files(pdf_files)

            if len(chunks_df) == 0:
                logger.warning("No chunks extracted from PDFs, skipping indexing")
                return False

            # FAISS 인덱스 생성
            index = self.faiss_index.create_index(embeddings)

            # 인덱스 및 관련 데이터 저장
            metadata_df = pd.DataFrame(metadata)
            success = self.faiss_index.save_index(index, chunks_df, metadata_df)

            return success
        except Exception as e:
            logger.error(f"Error indexing PDF files: {e}")
            return False


    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        인덱싱된 PDF에서 쿼리와 관련된 내용 검색

        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 결과 개수

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        if not self.faiss_index.is_initialized():
            logger.warning("FAISS index is not initialized")
            return []

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.get_embedding(query)

            # 검색 실행
            results = self.faiss_index.get_search_results(query_embedding, top_k)

            return results
        except Exception as e:
            logger.error(f"Error searching with query '{query}': {e}")
            return []


    def search_file(self, file_content: bytes, filename: str, query: str, top_k: int = DEFAULT_TOP_K) -> List[
        Dict[str, Any]]:
        """
        특정 PDF 파일에서만 검색 (인덱싱하지 않음)

        Args:
            file_content (bytes): PDF 파일 내용
            filename (str): 파일 이름
            query (str): 검색 쿼리
            top_k (int): 반환할 결과 개수

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        try:
            # PDF 파일 처리
            chunks, embeddings = self.pdf_service.process_temp_pdf(file_content, filename)

            if not chunks:
                logger.warning(f"No text chunks extracted from file '{filename}'")
                return []

            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.get_embedding(query)

            # 임시 인덱스 생성
            dimension = embeddings.shape[1]
            temp_index = faiss.IndexFlatIP(dimension)  # 코사인 유사도 검색을 위한 내적 제품 인덱스
            temp_index.add(embeddings)

            # 검색 실행
            scores, indices = temp_index.search(np.array([query_embedding]), min(top_k, len(chunks)))

            # 결과 가공
            results = []
            for i, idx in enumerate(indices[0]):
                similarity = float(scores[0][i])
                results.append({
                    "similarity": similarity,
                    **chunks[idx]
                })

            return results
        except Exception as e:
            logger.error(f"Error searching in file '{filename}' with query '{query}': {e}")
            return []


    def get_status(self) -> Dict[str, Any]:
        """
        검색 서비스 상태 정보 반환

        Returns:
            Dict[str, Any]: 상태 정보
        """
        index_stats = self.faiss_index.get_stats()

        if index_stats["status"] == "initialized":
            return {
                "status": "ready",
                "indexed_documents": index_stats.get("total_documents", 0),
                "indexed_chunks": index_stats.get("total_chunks", 0),
                "vector_dimension": index_stats.get("dimension", 0)
            }
        else:
            return {"status": "not_ready", "message": "검색 인덱스가 준비되지 않았습니다"}

