import { ref } from 'vue'
import { searchIndexed, searchInFile } from '@/api/search'
import { describeError } from '@/api/client'
import type { SearchResult } from '@/types/api'

export function useSearch() {
  const loading = ref(false)
  const results = ref<SearchResult[]>([])
  const message = ref<string | null>(null)
  const errorMessage = ref<string | null>(null)
  const lastQuery = ref('')

  async function run(query: string, topK: number, file: File | null) {
    if (!query.trim()) return
    loading.value = true
    errorMessage.value = null
    message.value = null
    lastQuery.value = query
    try {
      const data = file
        ? await searchInFile(file, query, topK)
        : await searchIndexed(query, topK)
      results.value = data.results
      message.value = data.message ?? null
    } catch (e) {
      errorMessage.value = describeError(e)
      results.value = []
    } finally {
      loading.value = false
    }
  }

  return { loading, results, message, errorMessage, lastQuery, run }
}
