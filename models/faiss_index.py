import os
import gc
import faiss
import logging
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Any, Optional

from config import INDEX_FILE_PATH, CHUNKS_FILE_PATH, METADATA_FILE_PATH

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS 인덱스를 관리하는 클래스
    코사인 유사도 기반 검색을 위한 IndexFlatIP를 사용
    CPU 전용으로 최적화
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FAISSIndex, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self):
        if self._initialized:
            return

        self.index = None
        self.chunks_df = None
        self.metadata_df = None
        self._initialized = True

        # 기존 인덱스가 있으면 로드
        self.load_if_exists()


    def load_if_exists(self) -> bool:
        """
        기존 인덱스 파일이 있으면 로드

        Returns:
            bool: 인덱스 로드 성공 여부
        """
        try:
            if os.path.exists(INDEX_FILE_PATH) and os.path.exists(CHUNKS_FILE_PATH):
                logger.info(f"Loading FAISS index from {INDEX_FILE_PATH}")
                self.index = faiss.read_index(INDEX_FILE_PATH)

                logger.info(f"Loading chunks data from {CHUNKS_FILE_PATH}")
                self.chunks_df = pd.read_csv(CHUNKS_FILE_PATH)

                # 메모리 최적화: 문자열 열을 카테고리 타입으로 변환
                for col in self.chunks_df.select_dtypes(include=['object']).columns:
                    self.chunks_df[col] = pd.Series(self.chunks_df[col], dtype='category')

                if os.path.exists(METADATA_FILE_PATH):
                    logger.info(f"Loading metadata from {METADATA_FILE_PATH}")
                    self.metadata_df = pd.read_csv(METADATA_FILE_PATH)

                    # 메모리 최적화: 문자열 열을 카테고리 타입으로 변환
                    for col in self.metadata_df.select_dtypes(include=['object']).columns:
                        self.metadata_df[col] = pd.Series(self.metadata_df[col], dtype='category')

                logger.info(f"FAISS index loaded with {len(self.chunks_df)} chunks")

                # 메모리 정리
                gc.collect()

                return True
            else:
                logger.info("No existing FAISS index found")
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False


    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        FAISS 인덱스 생성
        코사인 유사도를 위한 내적 제품(Inner Product) 인덱스 사용
        CPU 전용 최적화

        Args:
            embeddings (np.ndarray): 임베딩 벡터 배열

        Returns:
            faiss.Index: 생성된 FAISS 인덱스
        """
        if not isinstance(embeddings, np.ndarray) or len(embeddings) == 0:
            raise ValueError("Embeddings must be a non-empty numpy array")

        # 벡터 차원 확인
        dimension = embeddings.shape[1]
        num_vectors = embeddings.shape[0]

        logger.info(f"Creating FAISS index for {num_vectors} vectors of dimension {dimension}")

        # 메모리 사용량 최적화: 임베딩을 float32로 변환
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # IndexFlatIP는 내적 제품(Inner Product)을 사용한 인덱스로 정규화된 벡터 간의 내적은 코사인 유사도와 동일함
        # 주의: 이 방식은 벡터가 L2 정규화되어 있어야 함 (embedding.py에서 정규화 수행)
        index = faiss.IndexFlatIP(dimension)

        # CPU 사용 최적화: 대용량 데이터셋의 경우 배치 처리
        batch_size = 10000  # 이 값은 가용 메모리에 따라 조정

        if num_vectors > batch_size:
            logger.info(f"Adding vectors to index in batches of {batch_size}")
            for i in range(0, num_vectors, batch_size):
                end_idx = min(i + batch_size, num_vectors)
                index.add(embeddings[i:end_idx])
                logger.info(
                    f"Added batch {i // batch_size + 1}/{(num_vectors + batch_size - 1) // batch_size} to index")
                gc.collect()  # 메모리 정리
        else:
            # 임베딩 추가
            index.add(embeddings)

        logger.info(f"Created FAISS index with {index.ntotal} vectors")

        return index


    def save_index(self, index: faiss.Index, chunks_df: pd.DataFrame,
                   metadata_df: Optional[pd.DataFrame] = None) -> bool:
        """
        FAISS 인덱스 및 관련 데이터 저장

        Args:
            index (faiss.Index): 저장할 FAISS 인덱스
            chunks_df (pd.DataFrame): 청크 데이터
            metadata_df (pd.DataFrame, optional): 메타데이터

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(INDEX_FILE_PATH), exist_ok=True)

            logger.info(f"Saving FAISS index to {INDEX_FILE_PATH}")
            faiss.write_index(index, INDEX_FILE_PATH)

            logger.info(f"Saving chunks data to {CHUNKS_FILE_PATH}")
            chunks_df.to_csv(CHUNKS_FILE_PATH, index=False)

            # 메타데이터 저장
            if metadata_df is not None:
                logger.info(f"Saving metadata to {METADATA_FILE_PATH}")
                metadata_df.to_csv(METADATA_FILE_PATH, index=False)

            # 인스턴스 변수 업데이트
            self.index = index
            self.chunks_df = chunks_df
            self.metadata_df = metadata_df

            logger.info(f"Saved FAISS index with {len(chunks_df)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        FAISS 인덱스를 사용하여 쿼리 임베딩과 가장 유사한 벡터 검색

        Args:
            query_embedding (np.ndarray): 쿼리 임베딩 벡터
            top_k (int): 반환할 결과 개수

        Returns:
            Tuple[np.ndarray, np.ndarray]: (유사도 점수, 인덱스)
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized")

        # 쿼리 임베딩이 2D 배열이 아니면 변환
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # 메모리 사용량 최적화: 임베딩을 float32로 변환
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # 검색 실행
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        return scores, indices


    def get_search_results(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        검색 결과 데이터를 형식화하여 반환

        Args:
            query_embedding (np.ndarray): 쿼리 임베딩 벡터
            top_k (int): 반환할 결과 개수

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        if self.chunks_df is None:
            raise ValueError("Chunks data is not loaded")

        # 검색 실행
        scores, indices = self.search(query_embedding, top_k)

        # 검색 결과 가공
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.chunks_df) or idx < 0:
                continue  # 인덱스 범위 벗어나면 스킵

            chunk_data = self.chunks_df.iloc[idx].to_dict()

            # 유사도 점수 추가 (내적 값은 -1~1 범위이므로 0~1 범위로 정규화)
            similarity = float(scores[0][i])  # 이미 정규화된 벡터이므로 내적값이 코사인 유사도와 같음

            results.append({
                "similarity": similarity,
                **chunk_data
            })

        return results


    def is_initialized(self) -> bool:
        """
        인덱스가 초기화되었는지 확인

        Returns:
            bool: 인덱스 초기화 여부
        """
        return self.index is not None and self.chunks_df is not None


    def get_stats(self) -> Dict[str, Any]:
        """
        인덱스 통계 정보 반환

        Returns:
            Dict[str, Any]: 통계 정보
        """
        if not self.is_initialized():
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "total_chunks": len(self.chunks_df) if self.chunks_df is not None else 0,
            "total_documents": len(self.metadata_df) if self.metadata_df is not None else 0,
            "dimension": self.index.d if self.index is not None else 0
        }

