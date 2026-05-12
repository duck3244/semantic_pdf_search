import { ref } from 'vue'
import { uploadPdfs } from '@/api/upload'
import { describeError } from '@/api/client'
import type { UploadResponse } from '@/types/api'

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024 // backend MAX_UPLOAD_BYTES와 일치

export type UploadPhase = 'idle' | 'uploading' | 'indexing' | 'done' | 'error'

export function useUpload() {
  const phase = ref<UploadPhase>('idle')
  const percent = ref(0)
  const lastResult = ref<UploadResponse | null>(null)
  const errorMessage = ref<string | null>(null)

  function validate(files: File[]): { ok: File[]; rejected: string[] } {
    const ok: File[] = []
    const rejected: string[] = []
    for (const f of files) {
      const isPdf = f.name.toLowerCase().endsWith('.pdf')
      const okSize = f.size <= MAX_UPLOAD_BYTES
      if (isPdf && okSize) ok.push(f)
      else rejected.push(f.name + (!isPdf ? ' (PDF 아님)' : '') + (!okSize ? ' (100MB 초과)' : ''))
    }
    return { ok, rejected }
  }

  async function upload(files: File[]) {
    if (phase.value === 'uploading' || phase.value === 'indexing') return // 중복 클릭 가드
    errorMessage.value = null
    lastResult.value = null
    percent.value = 0
    phase.value = 'uploading'
    try {
      const result = await uploadPdfs(files, (p) => {
        percent.value = p
        if (p >= 100 && phase.value === 'uploading') phase.value = 'indexing'
      })
      lastResult.value = result
      phase.value = 'done'
    } catch (e) {
      errorMessage.value = describeError(e)
      phase.value = 'error'
    }
  }

  function reset() {
    phase.value = 'idle'
    percent.value = 0
    lastResult.value = null
    errorMessage.value = null
  }

  return { phase, percent, lastResult, errorMessage, validate, upload, reset }
}
