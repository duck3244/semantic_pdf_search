import { http } from './client'
import type { SearchResponse, SearchFileResponse, StatusResponse } from '@/types/api'

export async function getStatus(): Promise<StatusResponse> {
  const { data } = await http.get<StatusResponse>('/search/status/')
  return data
}

export async function searchIndexed(query: string, topK: number): Promise<SearchResponse> {
  const { data } = await http.get<SearchResponse>('/search/', {
    params: { query, top_k: topK },
  })
  return data
}

export async function searchInFile(
  file: File,
  query: string,
  topK: number,
  onProgress?: (percent: number) => void,
): Promise<SearchFileResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('query', query)
  form.append('top_k', String(topK))
  const { data } = await http.post<SearchFileResponse>('/search/file/', form, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}
