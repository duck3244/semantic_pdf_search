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
    FAISS 인덱스 관리 (CPU, IndexFlatIP + L2 정규화로 코사인 유사도).
    청크/메타데이터는 parquet로 저장하여 특수문자 안전.
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

        self.index: Optional[faiss.Index] = None
        self.chunks_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self._initialized = True

        self.load_if_exists()


    @staticmethod
    def _read_df(path: str) -> Optional[pd.DataFrame]:
        """파일 확장자/존재 여부에 따라 parquet 또는 csv로 자동 로드."""
        if os.path.exists(path):
            return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        # 신/구 포맷 상호 폴백
        if path.endswith(".parquet"):
            legacy = path[:-len(".parquet")] + ".csv"
            if os.path.exists(legacy):
                logger.info(f"Falling back to legacy CSV at {legacy}")
                return pd.read_csv(legacy)
        return None


    def load_if_exists(self) -> bool:
        try:
            if not os.path.exists(INDEX_FILE_PATH):
                logger.info("No existing FAISS index found")
                return False

            chunks_df = self._read_df(CHUNKS_FILE_PATH)
            if chunks_df is None:
                logger.info("Index file present but chunks file missing; skipping load")
                return False

            logger.info(f"Loading FAISS index from {INDEX_FILE_PATH}")
            self.index = faiss.read_index(INDEX_FILE_PATH)
            self.chunks_df = chunks_df
            self.metadata_df = self._read_df(METADATA_FILE_PATH)

            logger.info(f"FAISS index loaded with {len(self.chunks_df)} chunks")
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False


    @staticmethod
    def _to_float32(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings


    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        if not isinstance(embeddings, np.ndarray) or len(embeddings) == 0:
            raise ValueError("Embeddings must be a non-empty numpy array")

        dimension = embeddings.shape[1]
        num_vectors = embeddings.shape[0]
        logger.info(f"Creating FAISS index for {num_vectors} vectors of dimension {dimension}")

        embeddings = self._to_float32(embeddings)
        index = faiss.IndexFlatIP(dimension)

        batch_size = 10000
        if num_vectors > batch_size:
            for i in range(0, num_vectors, batch_size):
                end_idx = min(i + batch_size, num_vectors)
                index.add(embeddings[i:end_idx])
                logger.info(
                    f"Added batch {i // batch_size + 1}/{(num_vectors + batch_size - 1) // batch_size} to index")
                gc.collect()
        else:
            index.add(embeddings)

        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index


    def add_to_index(
        self,
        embeddings: np.ndarray,
        chunks_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """기존 인덱스에 추가 (없으면 새로 생성)."""
        if embeddings is None or len(embeddings) == 0 or chunks_df is None or len(chunks_df) == 0:
            logger.warning("Nothing to add to index")
            return False

        embeddings = self._to_float32(embeddings)

        if self.index is None:
            self.index = self.create_index(embeddings)
            new_chunks = chunks_df.reset_index(drop=True)
            new_metadata = metadata_df.reset_index(drop=True) if metadata_df is not None else None
        else:
            if self.index.d != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dim mismatch: index={self.index.d}, new={embeddings.shape[1]}"
                )
            self.index.add(embeddings)

            new_chunks = pd.concat(
                [self.chunks_df, chunks_df], ignore_index=True
            ) if self.chunks_df is not None else chunks_df.reset_index(drop=True)

            if metadata_df is not None:
                new_metadata = pd.concat(
                    [self.metadata_df, metadata_df], ignore_index=True
                ) if self.metadata_df is not None else metadata_df.reset_index(drop=True)
            else:
                new_metadata = self.metadata_df

        return self.save_index(self.index, new_chunks, new_metadata)


    def save_index(
        self,
        index: faiss.Index,
        chunks_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        try:
            os.makedirs(os.path.dirname(INDEX_FILE_PATH), exist_ok=True)

            logger.info(f"Saving FAISS index to {INDEX_FILE_PATH}")
            faiss.write_index(index, INDEX_FILE_PATH)

            logger.info(f"Saving chunks data to {CHUNKS_FILE_PATH}")
            chunks_df.to_parquet(CHUNKS_FILE_PATH, index=False)

            if metadata_df is not None:
                logger.info(f"Saving metadata to {METADATA_FILE_PATH}")
                metadata_df.to_parquet(METADATA_FILE_PATH, index=False)

            self.index = index
            self.chunks_df = chunks_df
            self.metadata_df = metadata_df

            logger.info(f"Saved FAISS index with {len(chunks_df)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("FAISS index is not initialized")

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        query_embedding = self._to_float32(query_embedding)

        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        return scores, indices


    def get_search_results(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.chunks_df is None:
            raise ValueError("Chunks data is not loaded")

        scores, indices = self.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks_df):
                continue
            chunk_data = self.chunks_df.iloc[idx].to_dict()
            results.append({
                "similarity": float(scores[0][i]),
                **chunk_data,
            })
        return results


    def is_initialized(self) -> bool:
        return self.index is not None and self.chunks_df is not None


    def get_stats(self) -> Dict[str, Any]:
        if not self.is_initialized():
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "total_chunks": len(self.chunks_df) if self.chunks_df is not None else 0,
            "total_documents": len(self.metadata_df) if self.metadata_df is not None else 0,
            "dimension": self.index.d if self.index is not None else 0,
        }
