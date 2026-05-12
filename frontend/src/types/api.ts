// 백엔드 응답 스키마 (backend/routers/*, backend/services/*와 1:1)

export interface SearchResult {
  similarity: number
  chunk_id: string
  text: string
  page: number
  paragraph: number
  filename: string
  path: string
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
  message?: string
}

export interface SearchFileResponse extends SearchResponse {
  filename: string
}

export type StatusResponse =
  | {
      status: 'ready'
      indexed_documents: number
      indexed_chunks: number
      vector_dimension: number
    }
  | {
      status: 'not_ready'
      message: string
    }

export interface UploadResponse {
  status: 'success'
  message: string
  indexed_files: string[]
  skipped_files: string[]
}

export interface ApiErrorBody {
  detail?: string
  message?: string
}
