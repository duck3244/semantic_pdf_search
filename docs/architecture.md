# 아키텍처 문서 (Architecture)

본 문서는 `semantic_pdf_search` 프로젝트의 시스템 구성, 계층 구조, 데이터 흐름, 그리고 주요 설계 결정을 설명합니다.

---

## 1. 개요

`semantic_pdf_search`는 한국어 PDF 문서에 대해 의미 기반(semantic) 검색을 제공하는 풀스택 애플리케이션입니다.

- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (Sentence-Transformers 호환, 한국어 특화)
- **벡터 검색**: FAISS `IndexFlatIP` + L2 정규화 → 코사인 유사도
- **실행 환경**: CPU 전용 최적화 (GPU 미사용)
- **백엔드**: FastAPI (Python)
- **프론트엔드**: Vue 3 + TypeScript + Vite + TailwindCSS

---

## 2. 컴포넌트 구성

```
┌─────────────────────────────────────────────────────────────┐
│                          Browser                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Vue 3 SPA (frontend/)                              │    │
│  │  ├─ Views      : HomeView                           │    │
│  │  ├─ Components : UploadZone / SearchPanel /         │    │
│  │  │              StatusBadge / ResultList /          │    │
│  │  │              ResultItem                          │    │
│  │  ├─ Composables: useUpload / useSearch              │    │
│  │  └─ API client : axios (baseURL = /api or '')       │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (JSON / multipart)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI (backend/main.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ CORSMiddleware│ │ Routers       │  │ SPA Static Mount│   │
│  │              │  │  /upload/*    │  │  (frontend/dist)│   │
│  │              │  │  /search/*    │  │                 │   │
│  └──────────────┘  └──────┬───────┘  └─────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Services                                           │     │
│  │  ├─ PDFService   : PDF 추출/저장/청크 분할         │     │
│  │  └─ SearchService: 인덱싱 흐름·쿼리 흐름 조립      │     │
│  └────────────────────────────────────────────────────┘     │
│                           │                                 │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Models (싱글톤)                                    │     │
│  │  ├─ EmbeddingModel : ko-sroberta-multitask (CPU)   │     │
│  │  └─ FAISSIndex     : IndexFlatIP + parquet 메타    │     │
│  └────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   Persistent Storage (backend/data/)                        │
│   ├─ pdfs/                : 업로드된 원본 PDF                │
│   └─ index/                                                 │
│      ├─ pdf_index.faiss   : FAISS 벡터 인덱스               │
│      ├─ pdf_chunks.parquet: 청크 텍스트/페이지/단락         │
│      └─ pdf_metadata.parquet: 파일별 메타데이터             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 계층(Layered) 구조

백엔드는 전형적인 4-계층 구조를 따릅니다.

| 계층 | 위치 | 책임 |
|---|---|---|
| **API (Router)** | `backend/routers/` | HTTP 입력 검증, 응답 직렬화, 상태 코드 매핑 |
| **Service** | `backend/services/` | 업무 로직 조립 (PDF → 청크 → 임베딩 → 인덱싱) |
| **Model** | `backend/models/` | 모델·인덱스 자원 보유 및 저수준 연산 |
| **Config / Infra** | `backend/config.py`, `backend/main.py` | 환경설정, 라이프사이클, 정적 마운트, CORS |

프론트엔드는 컴포넌트 + 상태(composable) + API 클라이언트로 분리되어 있습니다.

| 계층 | 위치 | 책임 |
|---|---|---|
| **View** | `frontend/src/views/` | 페이지 컴포지션 (HomeView) |
| **Component** | `frontend/src/components/` | 화면 단위 위젯 |
| **Composable** | `frontend/src/composables/` | 도메인 상태 + 비동기 흐름 캡슐화 |
| **API** | `frontend/src/api/` | 백엔드 엔드포인트별 래퍼 |
| **Types** | `frontend/src/types/` | 응답 스키마 타입 정의 |

---

## 4. 주요 모듈 책임

### 4.1 Backend

- **`main.py`**
  - FastAPI 앱 생성, CORS, 라우터 등록
  - `lifespan`에서 임베딩 모델 warm-up (cold start 제거)
  - `FRONTEND_DIST`가 존재하면 SPA 정적 마운트, 없으면 API-only 안내 페이지

- **`config.py`**
  - 경로/모델/배치 크기/업로드 한계 등의 상수
  - 로컬 모델 스냅샷 존재 시 로컬 경로, 없으면 HF Hub ID로 폴백
  - `torch.set_num_threads(min(cpu_count, 4))`로 CPU 스레드 제한

- **`models/embedding.py`**
  - `EmbeddingModel` 싱글톤
  - Mean pooling + L2 normalize → Sentence-Transformers 호환 임베딩
  - `EMBEDDING_BATCH_SIZE`(기본 8) 단위 배치 forward
  - `use_safetensors=True`로 안전한 가중치 로드 (CVE-2025-32434 회피)

- **`models/faiss_index.py`**
  - `FAISSIndex` 싱글톤
  - `IndexFlatIP` (내적) + 정규화 → 코사인 유사도와 동치
  - 청크/메타데이터는 parquet 저장 (특수문자 안전, CSV 폴백 지원)
  - `add_to_index`로 누적 추가, `save_index`/`load_if_exists`로 영속화

- **`services/pdf_service.py`**
  - PyMuPDF(`fitz`)로 페이지 단위 텍스트 추출
  - 빈 줄(이중 개행) 기준 단락 분할, 20자 미만 청크 제외
  - 업로드 저장 시 path traversal 방지(basename + PDF_DIR 하위 재검증)
  - 임시 PDF용 처리 경로(`process_temp_pdf`) 제공

- **`services/search_service.py`**
  - `index_pdf_files`: PDFService → 청크/임베딩 → FAISSIndex 누적
  - `search`: 인덱싱된 코퍼스 전체 검색
  - `search_file`: 인덱스 추가 없이 단일 PDF 임시 검색
  - `get_status`: 인덱스 상태 요약

- **`routers/upload.py`** — `POST /upload/pdf/` (다중 파일)
- **`routers/search.py`** — `GET /search/`, `POST /search/file/`, `GET /search/status/`

### 4.2 Frontend

- **`api/client.ts`** — axios 인스턴스, `baseURL` 분기, 에러 메시지 변환
- **`api/upload.ts`** — `POST /upload/pdf/` 멀티파트
- **`api/search.ts`** — `GET /search/`, `POST /search/file/`, `GET /search/status/`
- **`composables/useUpload.ts`** — 업로드 상태머신: `idle → uploading → indexing → done|error`
- **`composables/useSearch.ts`** — 검색 결과·로딩·에러 상태 관리
- **`components/UploadZone.vue`** — 드래그&드롭/클릭 업로드, 진행률 표시
- **`components/SearchPanel.vue`** — 쿼리/Top-K/단일 파일 모드 UI
- **`components/ResultList.vue` / `ResultItem.vue`** — 결과 목록 + 키워드 하이라이트
- **`components/StatusBadge.vue`** — 인덱스 상태 배지

---

## 5. 데이터 흐름

### 5.1 인덱싱 흐름 (Upload → Index)

1. 사용자가 PDF를 드래그&드롭 또는 선택 → `UploadZone.vue`
2. `useUpload.upload()` → `api/upload.uploadPdfs()` → `POST /upload/pdf/`
3. `routers/upload.upload_pdfs`:
   - 확장자/크기(`MAX_UPLOAD_BYTES = 100MB`) 검증
   - `PDFService.save_uploaded_pdf()`로 디스크에 저장 (path traversal 차단)
4. `SearchService.index_pdf_files()`:
   - `PDFService.process_pdf_files()` → 텍스트 추출 → 단락 청크 → 임베딩 배치 생성
   - `FAISSIndex.add_to_index()` → 벡터 누적 → `save_index()`로 영속화
5. 응답으로 인덱싱/건너뜀 파일 목록 반환 → `StatusBadge` 새로고침

### 5.2 코퍼스 검색 흐름 (Search)

1. `SearchPanel.vue`에서 쿼리 입력 → `useSearch.run()`
2. `api/search.searchIndexed()` → `GET /search/?query=...&top_k=...`
3. `routers/search.search_pdfs`:
   - `SearchService.get_status()`로 인덱스 준비 확인
   - `EmbeddingModel.get_embedding(query)` → L2 정규화된 쿼리 벡터
   - `FAISSIndex.get_search_results()` → 상위 K개 청크 + 유사도(내적값)
4. 응답: `[{ similarity, chunk_id, text, page, paragraph, filename, path }, ...]`
5. `ResultList`/`ResultItem`이 결과 + 쿼리 단어 강조 렌더링

### 5.3 단일 파일 임시 검색 흐름 (Search-in-file)

1. `SearchPanel`에서 "단일 PDF" 모드 + 파일 선택 → `useSearch.run(file)`
2. `POST /search/file/` (multipart: file, query, top_k)
3. `SearchService.search_file()`:
   - `PDFService.process_temp_pdf()`로 tempdir에 임시 저장 → 추출 → 임베딩
   - 임시 `IndexFlatIP` 생성·검색 → 결과 반환 후 tempdir 삭제
4. **인덱스에 추가되지 않음** (휘발성)

---

## 6. 영속성 (Persistence)

| 자원 | 경로 | 포맷 | 비고 |
|---|---|---|---|
| 원본 PDF | `backend/data/pdfs/` | binary | 업로드 시 basename으로 정규화, 충돌 시 8자 hex suffix |
| FAISS 인덱스 | `backend/data/index/pdf_index.faiss` | FAISS binary | `IndexFlatIP` (float32) |
| 청크 | `backend/data/index/pdf_chunks.parquet` | Parquet | columns: `chunk_id, text, page, paragraph, filename, path` |
| 메타데이터 | `backend/data/index/pdf_metadata.parquet` | Parquet | columns: `filename, path, file_size, total_chunks` |
| 레거시 폴백 | `*.csv` | CSV | parquet 부재 시 동명 csv를 자동으로 로드 |

---

## 7. 주요 설계 결정 및 근거

- **CPU 전용** — GPU가 없는 환경에서도 동작하도록 `CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(<=4)` 강제. FAISS도 CPU 빌드(`faiss-cpu`).
- **IndexFlatIP + 정규화** — 코사인 유사도는 정규화된 벡터의 내적과 동치. 정확도 손실 없는 brute-force가 코퍼스 규모에 비해 충분히 빠르고 구현이 단순.
- **Parquet 저장** — 한국어 파일명·줄바꿈 등 특수문자에 강하고 압축률·로드 속도 우수. CSV는 구포맷 호환용 폴백.
- **싱글톤(EmbeddingModel, FAISSIndex)** — 워커당 1회 초기화로 모델/인덱스 메모리 중복 적재 방지.
- **lifespan warm-up** — 첫 요청의 ~10초 cold start를 startup 시점으로 이동.
- **Vue Router hash 모드** — FastAPI `StaticFiles(html=True)` 마운트와 충돌 없이 클라이언트 라우팅 동작 (서버 측 fallback 불필요).
- **dev 시 Vite proxy(`/api` → `:8000`)** — CORS/Origin 이슈 회피.
- **운영 시 단일 origin** — `frontend/dist`를 FastAPI가 `/`에 마운트해 API와 SPA를 같은 origin으로 서빙.
- **업로드 한계 100MB** — 단일 파일 메모리 사용 상한. 초과 시 `skipped_files`로 응답.
- **path traversal 방어** — `basename()` 정규화 + `PDF_DIR` 하위 경로 재검증.

---

## 8. 배포 토폴로지

- **개발**: `make dev` — 백엔드 `:8000`(uvicorn reload), 프론트 `:5173`(Vite). 프론트의 `/api/*`가 백엔드로 프록시됨.
- **운영**: `make build` → `frontend/dist` 생성 → 백엔드 단독 기동 시 `/` 에서 SPA, `/upload/*`, `/search/*` 에서 API 서빙. (단일 프로세스, 단일 origin)

---

## 9. 보안·운영 고려

- CORS 화이트리스트는 `ALLOWED_ORIGINS` 환경변수로 좁힐 수 있으며 기본은 `*` (credentials 미사용).
- `pytorch_model.bin` 차단(CVE-2025-32434) 회피 위해 `use_safetensors=True` 강제.
- 전역 예외 핸들러가 500 응답을 일관 포맷으로 변환 (`{"message": ...}`).
- 로그는 stdout + `backend/app.log`로 동시 기록. uvicorn reload 시 `*.log`, `data/*`는 watcher에서 제외하여 자기참조 reload 루프를 방지.
