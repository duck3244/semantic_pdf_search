import os
import torch
import logging
import numpy as np

from transformers import AutoTokenizer, AutoModel
from typing import Union, List

from config import EMBEDDING_MODEL_NAME, MAX_LENGTH, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    텍스트 임베딩을 위한 모델 클래스
    ko-sroberta-multitask 모델을 사용하여 텍스트 임베딩 생성
    CPU 전용으로 설정
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

        # 명시적으로 CUDA 비활성화
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} (CPU only)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

            # 모델이 GPU에 로드되어 있으면 CPU로 이동
            self.model = self.model.cpu()

            # 모델 평가 모드로 설정 (학습 모드 비활성화)
            self.model.eval()

            self._initialized = True
            logger.info("Embedding model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise


    def get_embedding(self, text: str) -> np.ndarray:
        """
        텍스트에서 임베딩을 추출

        Args:
            text (str): 임베딩을 생성할 텍스트

        Returns:
            np.ndarray: 생성된 텍스트 임베딩 벡터
        """
        # 입력 텍스트 전처리
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )

        # CPU에서 모델 실행
        with torch.no_grad():
            # 입력 텐서를 CPU로 강제 이동
            inputs = {k: v.cpu() for k, v in inputs.items()}
            outputs = self.model(**inputs)

        # [CLS] 토큰의 임베딩을 사용 (문장의 의미 표현)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # 정규화 - Cosine 유사도 검색을 위해 중요
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy()


    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트에서 임베딩을 추출

        Args:
            texts (List[str]): 임베딩을 생성할 텍스트 리스트

        Returns:
            np.ndarray: 생성된 텍스트 임베딩 벡터 배열
        """
        embeddings = []

        # 설정 파일에서 배치 크기 가져옴 (메모리 사용량 제어)
        batch_size = EMBEDDING_BATCH_SIZE

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            logger.info(f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch_texts)} texts)")

            # 각 텍스트에 대해 임베딩 생성
            for text in batch_texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)

            # 명시적으로 메모리 정리 (필요시)
            if i % (batch_size * 5) == 0 and i > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return np.array(embeddings).astype('float32')

