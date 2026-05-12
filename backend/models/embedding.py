import os
import torch
import logging
import numpy as np

from transformers import AutoTokenizer, AutoModel
from typing import List

from config import EMBEDDING_MODEL_NAME, MAX_LENGTH, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    ko-sroberta-multitask 기반 문장 임베딩 (CPU 전용).
    sentence-transformers 모델이므로 mean pooling + L2 normalize 사용.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self):
        if self._initialized:
            return

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} (CPU only)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            # torch<2.6 + transformers>=4.45 환경에서 pytorch_model.bin 로드가 차단됨
            # (CVE-2025-32434). safetensors가 있으면 그것을 강제해 안전하게 로드.
            self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, use_safetensors=True)
            self.model = self.model.cpu()
            self.model.eval()

            self._initialized = True
            logger.info("Embedding model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise


    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # sentence-transformers 정식 풀링: 토큰 임베딩을 attention mask로 가중 평균
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        pooled = self._mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype('float32')


    def get_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩."""
        return self._encode_batch([text])[0]


    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """여러 텍스트의 임베딩을 배치 forward로 계산."""
        if not texts:
            return np.empty((0, 0), dtype='float32')

        batch_size = EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch_texts)} texts)")
            all_embeddings.append(self._encode_batch(batch_texts))

        return np.vstack(all_embeddings)
