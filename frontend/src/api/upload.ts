import { http } from './client'
import type { UploadResponse } from '@/types/api'

export async function uploadPdfs(
  files: File[],
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const { data } = await http.post<UploadResponse>('/upload/pdf/', form, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}
