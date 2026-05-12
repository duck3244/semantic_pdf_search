import axios, { AxiosError } from 'axios'
import type { ApiErrorBody } from '@/types/api'

// 개발: Vite proxy가 /api/* → backend(:8000)으로 포워딩
// 운영: SPA가 backend와 같은 origin에서 서빙되므로 prefix 불필요
// 명시적 오버라이드가 필요하면 VITE_API_BASE_URL 사용
const baseURL =
  import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? '/api' : '')

export const http = axios.create({
  baseURL,
  // 인덱싱이 동기 처리라 큰 PDF에서 분 단위 소요 가능 → 10분
  timeout: 10 * 60 * 1000,
})

export function describeError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const e = err as AxiosError<ApiErrorBody>
    const body = e.response?.data
    return body?.detail || body?.message || e.message
  }
  if (err instanceof Error) return err.message
  return String(err)
}
