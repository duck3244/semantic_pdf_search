# UML 다이어그램

본 문서는 `semantic_pdf_search`의 구조와 동작을 Mermaid 기반 UML로 시각화합니다.
모든 다이어그램은 GitHub Mermaid 렌더링과 호환됩니다.

목차
1. [컴포넌트 다이어그램](#1-컴포넌트-다이어그램-component-diagram)
2. [백엔드 클래스 다이어그램](#2-백엔드-클래스-다이어그램-class-diagram)
3. [프론트엔드 모듈 다이어그램](#3-프론트엔드-모듈-다이어그램)
4. [업로드 & 인덱싱 시퀀스](#4-업로드--인덱싱-시퀀스-sequence-diagram)
5. [코퍼스 검색 시퀀스](#5-코퍼스-검색-시퀀스-sequence-diagram)
6. [단일 파일 임시 검색 시퀀스](#6-단일-파일-임시-검색-시퀀스-sequence-diagram)
7. [업로드 상태머신](#7-업로드-상태머신-state-diagram)
8. [데이터 흐름](#8-데이터-흐름-data-flow)
9. [배포 다이어그램](#9-배포-다이어그램-deployment-diagram)

---

## 1. 컴포넌트 다이어그램 (Component Diagram)

```mermaid
flowchart LR
  subgraph Browser["Browser (SPA)"]
    direction TB
    Views["views/HomeView"]
    Comps["components<br/>UploadZone · SearchPanel<br/>ResultList · ResultItem · StatusBadge"]
    Composables["composables<br/>useUpload · useSearch"]
    APIClient["api/<br/>client · upload · search"]
    Views --> Comps
    Comps --> Composables
    Composables --> APIClient
  end

  subgraph FastAPI["FastAPI Backend"]
    direction TB
    Routers["routers<br/>upload · search"]
    Services["services<br/>PDFService · SearchService"]
    Models["models<br/>EmbeddingModel · FAISSIndex"]
    Routers --> Services
    Services --> Models
  end

  subgraph Storage["Persistent Storage (backend/data)"]
    PDFs[("pdfs/*.pdf")]
    Idx[("index/pdf_index.faiss")]
    Chunks[("index/pdf_chunks.parquet")]
    Meta[("index/pdf_metadata.parquet")]
  end

  APIClient -- "HTTP /upload, /search" --> Routers
  Models --> Idx
  Models --> Chunks
  Models --> Meta
  Services --> PDFs

  classDef ext fill:#fff3e0,stroke:#fb8c00
  HF["HuggingFace Hub<br/>jhgan/ko-sroberta-multitask"]:::ext
  Models -. "최초 1회 가중치 다운로드" .-> HF
```

---

## 2. 백엔드 클래스 다이어그램 (Class Diagram)

```mermaid
classDiagram
  direction LR

  class EmbeddingModel {
    <<singleton>>
    -tokenizer
    -model
    -_initialized: bool
    +get_embedding(text: str) ndarray
    +get_embeddings(texts: List[str]) ndarray
    -_encode_batch(texts: List[str]) ndarray
    -_mean_pool(hidden, mask) Tensor
  }

  class FAISSIndex {
    <<singleton>>
    -index: faiss.Index
    -chunks_df: DataFrame
    -metadata_df: DataFrame
    +load_if_exists() bool
    +create_index(embeddings) faiss.Index
    +add_to_index(embeddings, chunks_df, metadata_df) bool
    +save_index(index, chunks_df, metadata_df) bool
    +search(query_embedding, top_k) Tuple
    +get_search_results(query_embedding, top_k) List~Dict~
    +is_initialized() bool
    +get_stats() Dict
  }

  class PDFService {
    -embedding_model: EmbeddingModel
    +extract_text_from_pdf(pdf_path) List~Dict~
    +process_pdf_files(pdf_files) Tuple
    +save_uploaded_pdf(file_content, filename) str
    +process_temp_pdf(file_content, filename) Tuple
  }

  class SearchService {
    -embedding_model: EmbeddingModel
    -faiss_index: FAISSIndex
    -pdf_service: PDFService
    +index_pdf_files(pdf_files) bool
    +search(query, top_k) List~Dict~
    +search_file(file_content, filename, query, top_k) List~Dict~
    +get_status() Dict
  }

  class UploadRouter {
    <<FastAPI router>>
    +POST /upload/pdf/(files)
  }

  class SearchRouter {
    <<FastAPI router>>
    +GET  /search/(query, top_k)
    +POST /search/file/(file, query, top_k)
    +GET  /search/status/()
  }

  class FastAPIApp {
    +lifespan(app)
    +exception_handler(exc)
  }

  FastAPIApp --> UploadRouter : include_router
  FastAPIApp --> SearchRouter : include_router
  UploadRouter --> PDFService : use
  UploadRouter --> SearchService : use
  SearchRouter --> SearchService : use
  SearchService --> PDFService : use
  SearchService --> EmbeddingModel : use
  SearchService --> FAISSIndex : use
  PDFService --> EmbeddingModel : use
  class Transformers {
    <<library>>
    AutoModel
    AutoTokenizer
  }
  class Faiss {
    <<library>>
    IndexFlatIP
  }
  EmbeddingModel ..> Transformers : uses
  FAISSIndex ..> Faiss : uses
```

---

## 3. 프론트엔드 모듈 다이어그램

```mermaid
classDiagram
  direction TB

  class HomeView {
    +onIndexed()
  }

  class UploadZone {
    <<component>>
    +emits: indexed
    -handle(files)
  }
  class SearchPanel {
    <<component>>
    -query: string
    -topK: number
    -fileMode: bool
    -file: File
    -onSubmit()
  }
  class StatusBadge {
    <<component>>
    +refresh()
  }
  class ResultList {
    <<component>>
    +props: results, query, loading, message, errorMessage
  }
  class ResultItem {
    <<component>>
    +props: result, query
    -segments(): Segment[]
  }

  class useUpload {
    <<composable>>
    +phase: UploadPhase
    +percent: number
    +lastResult: UploadResponse
    +validate(files)
    +upload(files)
    +reset()
  }
  class useSearch {
    <<composable>>
    +loading: bool
    +results: SearchResult[]
    +errorMessage: string
    +run(query, topK, file)
  }

  class apiUpload {
    <<api>>
    +uploadPdfs(files, onProgress)
  }
  class apiSearch {
    <<api>>
    +getStatus()
    +searchIndexed(query, topK)
    +searchInFile(file, query, topK, onProgress)
  }
  class httpClient {
    <<axios>>
    +http
    +describeError(err)
  }

  HomeView --> UploadZone
  HomeView --> SearchPanel
  HomeView --> StatusBadge
  SearchPanel --> ResultList
  ResultList --> ResultItem

  UploadZone --> useUpload
  SearchPanel --> useSearch
  StatusBadge --> apiSearch

  useUpload --> apiUpload
  useSearch --> apiSearch
  apiUpload --> httpClient
  apiSearch --> httpClient
```

---

## 4. 업로드 & 인덱싱 시퀀스 (Sequence Diagram)

```mermaid
sequenceDiagram
  autonumber
  actor U as 사용자
  participant UZ as UploadZone.vue
  participant CU as useUpload
  participant AX as axios (http)
  participant R as UploadRouter
  participant SS as SearchService
  participant PS as PDFService
  participant EM as EmbeddingModel
  participant FI as FAISSIndex
  participant FS as Filesystem

  U->>UZ: PDF 드롭/선택
  UZ->>CU: upload(files)
  CU->>AX: POST /upload/pdf/ (multipart)
  AX->>R: HTTP request
  loop 각 파일
    R->>R: 확장자 / 크기(100MB) 검증
    R->>PS: save_uploaded_pdf(content, name)
    PS->>FS: write data/pdfs/<safe>.pdf
  end
  R->>SS: index_pdf_files(saved_paths)
  SS->>PS: process_pdf_files(paths)
  loop 각 PDF
    PS->>PS: PyMuPDF 텍스트 추출 + 단락 청크
  end
  PS->>EM: get_embeddings(texts) [배치 8]
  EM-->>PS: embeddings (float32, normalized)
  PS-->>SS: (chunks_df, embeddings, metadata)
  SS->>FI: add_to_index(embeddings, chunks_df, metadata_df)
  FI->>FS: faiss.write_index + parquet 저장
  FI-->>SS: True
  SS-->>R: True
  R-->>AX: 200 { indexed_files, skipped_files }
  AX-->>CU: result
  CU-->>UZ: phase = 'done'
  UZ-->>U: 완료 표시 (+ StatusBadge.refresh)
```

---

## 5. 코퍼스 검색 시퀀스 (Sequence Diagram)

```mermaid
sequenceDiagram
  autonumber
  actor U as 사용자
  participant SP as SearchPanel.vue
  participant CS as useSearch
  participant AX as axios (http)
  participant R as SearchRouter
  participant SS as SearchService
  participant EM as EmbeddingModel
  participant FI as FAISSIndex

  U->>SP: 쿼리 입력 + 검색 클릭
  SP->>CS: run(query, topK, null)
  CS->>AX: GET /search/?query=&top_k=
  AX->>R: HTTP request
  R->>SS: get_status()
  alt index not_ready
    SS-->>R: status=not_ready
    R-->>AX: 400 인덱스 준비 안 됨
  else ready
    R->>SS: search(query, top_k)
    SS->>EM: get_embedding(query)
    EM-->>SS: query_vec (normalized)
    SS->>FI: get_search_results(query_vec, top_k)
    FI->>FI: IndexFlatIP.search → (scores, idxs)
    FI-->>SS: results[{ similarity, chunk, ... }]
    SS-->>R: results
    R-->>AX: 200 { query, results }
  end
  AX-->>CS: data
  CS-->>SP: results / message / error
  SP-->>U: ResultList 렌더링 (키워드 강조)
```

---

## 6. 단일 파일 임시 검색 시퀀스 (Sequence Diagram)

```mermaid
sequenceDiagram
  autonumber
  actor U as 사용자
  participant SP as SearchPanel.vue
  participant CS as useSearch
  participant AX as axios (http)
  participant R as SearchRouter
  participant SS as SearchService
  participant PS as PDFService
  participant EM as EmbeddingModel
  participant TF as Temp FAISS Index

  U->>SP: 단일 PDF + 쿼리 + 검색
  SP->>CS: run(query, topK, file)
  CS->>AX: POST /search/file/ (multipart)
  AX->>R: HTTP request
  R->>R: 확장자 검증
  R->>SS: search_file(content, filename, query, top_k)
  SS->>PS: process_temp_pdf(content, filename)
  PS->>PS: tempdir 저장 → 추출 → 청크
  PS->>EM: get_embeddings(chunk_texts)
  EM-->>PS: chunk_vecs
  PS-->>SS: (chunks, chunk_vecs)
  SS->>EM: get_embedding(query)
  EM-->>SS: query_vec
  SS->>TF: IndexFlatIP(dim).add(chunk_vecs)
  SS->>TF: search(query_vec, top_k)
  TF-->>SS: scores, idxs
  SS-->>R: results
  R-->>AX: 200 { query, filename, results }
  AX-->>CS: data
  CS-->>SP: results
  SP-->>U: ResultList 렌더링
  Note over PS: tempdir 삭제 (휘발성, 인덱스에 추가되지 않음)
```

---

## 7. 업로드 상태머신 (State Diagram)

`useUpload`의 `phase` 변수가 따르는 상태 전이입니다.

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> uploading: upload(files)
  uploading --> indexing: 진행률 100%
  uploading --> error: 네트워크/검증 실패
  indexing --> done: 서버 200 응답
  indexing --> error: 서버 5xx / 4xx
  done --> idle: reset()
  error --> idle: reset()
  done --> [*]
```

---

## 8. 데이터 흐름 (Data Flow)

```mermaid
flowchart TB
  PDF[("원본 PDF<br/>data/pdfs/")] -->|PyMuPDF.get_text| RAW["페이지 텍스트"]
  RAW -->|이중 개행 분할 · 20자 미만 제외| CH["청크<br/>chunk_id · text · page · paragraph"]
  CH -->|tokenizer + mean pool + L2 norm| EMB["임베딩 float32 dim≈768"]
  CH -->|to_parquet| CHP[("pdf_chunks.parquet")]
  EMB -->|IndexFlatIP.add| IDX[("pdf_index.faiss")]
  META["파일별 메타<br/>filename · file_size · total_chunks"] -->|to_parquet| METP[("pdf_metadata.parquet")]

  Q["사용자 쿼리"] -->|동일 임베딩 파이프라인| QV["쿼리 벡터"]
  QV -->|IndexFlatIP.search| TOPK["top_k scores · idxs"]
  TOPK -->|chunks_df.iloc| RES["결과 객체<br/>similarity · text · page · filename ..."]
```

---

## 9. 배포 다이어그램 (Deployment Diagram)

```mermaid
flowchart LR
  subgraph Dev["개발 환경 (make dev)"]
    direction TB
    DV["Vite Dev Server<br/>localhost:5173"]
    DB["uvicorn (RELOAD=1)<br/>localhost:8000"]
    DV -- "/api/* → :8000<br/>(Vite proxy)" --> DB
  end

  subgraph Prod["운영 환경 (make build && python main.py)"]
    direction TB
    PB["FastAPI / uvicorn<br/>:8000"]
    SPA["StaticFiles 마운트<br/>frontend/dist (/)"]
    API["API 라우트<br/>/upload/* · /search/*"]
    PB --> SPA
    PB --> API
  end

  Client["Browser"] --> DV
  Client --> PB

  subgraph Disk["디스크 (호스트)"]
    P1[("backend/data/pdfs/")]
    P2[("backend/data/index/")]
  end
  DB --> P1
  DB --> P2
  PB --> P1
  PB --> P2
```
