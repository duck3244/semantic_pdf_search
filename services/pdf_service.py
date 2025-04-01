import os
import gc
import time
import shutil
import logging
import tempfile
import numpy as np
import pandas as pd

from PyPDF2 import PdfReader
from typing import Tuple, List, Dict, Any

from models.embedding import EmbeddingModel
from config import PDF_DIR, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class PDFService:
    """
    PDF 처리 및 임베딩 생성을 담당하는 서비스
    CPU 최적화를 위한 처리 추가
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()


    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDF 파일에서 텍스트 추출 및 청크로 분할

        Args:
            pdf_path (str): PDF 파일 경로

        Returns:
            List[Dict[str, Any]]: 추출된 텍스트 청크 리스트
        """
        try:
            reader = PdfReader(pdf_path)
            text_chunks = []

            filename = os.path.basename(pdf_path)
            total_pages = len(reader.pages)

            logger.info(f"Extracting text from PDF: {filename} ({total_pages} pages)")

            for i, page in enumerate(reader.pages):
                # 메모리 문제를 줄이기 위해 주기적으로 파이썬 가비지 컬렉션 강제 실행
                if i > 0 and i % 50 == 0:
                    gc.collect()
                    logger.info(f"Processed {i}/{total_pages} pages from {filename}")

                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # 텍스트를 문단 또는 일정 크기 청크로 분할
                paragraphs = text.split('\n\n')
                for j, para in enumerate(paragraphs):
                    if not para or not para.strip():
                        continue

                    # 너무 작은 청크는 건너뛰기 (선택사항: 성능 최적화를 위함)
                    if len(para.strip()) < 20:
                        continue

                    chunk_id = f"{filename}_p{i + 1}_c{j + 1}"
                    text_chunks.append({
                        'chunk_id': chunk_id,
                        'text': para.strip(),
                        'page': i + 1,
                        'paragraph': j + 1,
                        'filename': filename,
                        'path': pdf_path
                    })

            logger.info(f"Extracted {len(text_chunks)} text chunks from PDF: {filename}")
            return text_chunks
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise


    def process_pdf_files(self, pdf_files: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]:
        """
        여러 PDF 파일 처리 및 임베딩 생성
        CPU 최적화를 위한 배치 처리 적용

        Args:
            pdf_files (List[str]): 처리할 PDF 파일 경로 리스트

        Returns:
            Tuple[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]:
                (청크 데이터프레임, 임베딩 배열, 메타데이터 리스트)
        """
        all_chunks = []
        all_metadata = []

        for pdf_file in pdf_files:
            logger.info(f"Processing PDF file: {pdf_file}")
            start_time = time.time()

            # PDF 메타데이터 수집
            filename = os.path.basename(pdf_file)
            file_size = os.path.getsize(pdf_file)

            # PDF에서 텍스트 추출
            chunks = self.extract_text_from_pdf(pdf_file)
            all_chunks.extend(chunks)

            # 메타데이터 추가
            metadata = {
                'filename': filename,
                'path': pdf_file,
                'file_size': file_size,
                'total_chunks': len(chunks)
            }
            all_metadata.append(metadata)

            processing_time = time.time() - start_time
            logger.info(f"Finished processing {filename} in {processing_time:.2f} seconds")

            # 메모리 정리
            gc.collect()

        if not all_chunks:
            logger.warning("No text chunks extracted from PDF files")
            return pd.DataFrame(), np.array([]), all_metadata

        # 청크를 DataFrame으로 변환
        chunks_df = pd.DataFrame(all_chunks)

        # 메모리 최적화를 위해 객체 타입을 최적화
        for col in chunks_df.select_dtypes(include=['object']).columns:
            chunks_df[col] = pd.Series(chunks_df[col], dtype='category')

        # 각 청크에 대한 임베딩 생성 (배치 처리)
        logger.info(f"Generating embeddings for {len(chunks_df)} text chunks")
        start_time = time.time()

        texts = chunks_df['text'].tolist()
        batch_size = EMBEDDING_BATCH_SIZE

        # 배치 단위로 임베딩 생성
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + min(batch_size, len(texts) - i)]
            logger.info(
                f"Processing embeddings batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

            batch_embeddings = self.embedding_model.get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

            # 메모리 정리
            gc.collect()

        # 모든 배치의 임베딩을 하나의 배열로 병합
        embeddings = np.vstack(all_embeddings).astype('float32')

        processing_time = time.time() - start_time
        logger.info(f"Finished generating {len(embeddings)} embeddings in {processing_time:.2f} seconds")

        return chunks_df, embeddings, all_metadata


    def save_uploaded_pdf(self, file_content: bytes, filename: str) -> str:
        """
        업로드된 PDF 파일 저장

        Args:
            file_content (bytes): 파일 내용
            filename (str): 파일 이름

        Returns:
            str: 저장된 파일 경로
        """
        # 파일명 충돌 방지를 위한 임시 사본 처리
        if os.path.exists(os.path.join(PDF_DIR, filename)):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{os.urandom(4).hex()}{ext}"

        file_path = os.path.join(PDF_DIR, filename)

        with open(file_path, 'wb') as f:
            f.write(file_content)

        logger.info(f"Saved uploaded PDF: {filename}")
        return file_path


    def process_temp_pdf(self, file_content: bytes, filename: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        임시 PDF 파일 처리 및 임베딩 생성 (파일 업로드 없이 검색만 할 때 사용)
        CPU 최적화 적용

        Args:
            file_content (bytes): 파일 내용
            filename (str): 파일 이름

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray]: (청크 리스트, 임베딩 배열)
        """
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)

        try:
            # 임시 파일 저장
            with open(temp_path, 'wb') as f:
                f.write(file_content)

            # PDF에서 텍스트 추출
            chunks = self.extract_text_from_pdf(temp_path)

            if not chunks:
                return [], np.array([])

            # 각 청크에 대한 임베딩 생성
            logger.info(f"Generating embeddings for {len(chunks)} text chunks from temporary file")
            start_time = time.time()

            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.get_embeddings(texts)

            processing_time = time.time() - start_time
            logger.info(f"Finished generating embeddings in {processing_time:.2f} seconds")

            return chunks, embeddings
        finally:
            # 임시 디렉토리 삭제
            shutil.rmtree(temp_dir)

            # 메모리 정리
            gc.collect()

