<script setup lang="ts">
import ResultItem from './ResultItem.vue'
import type { SearchResult } from '@/types/api'

defineProps<{
  results: SearchResult[]
  query: string
  loading: boolean
  message: string | null
  errorMessage: string | null
}>()
</script>

<template>
  <section class="mt-4 flex flex-col gap-3">
    <div v-if="loading" class="rounded border border-slate-200 bg-white p-4 text-sm text-slate-500">
      검색 중…
    </div>
    <div v-else-if="errorMessage" class="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      검색 오류: {{ errorMessage }}
    </div>
    <template v-else>
      <div
        v-if="results.length"
        class="text-sm text-slate-500"
      >
        "<strong class="text-slate-900">{{ query }}</strong>" 결과 {{ results.length }}건
      </div>
      <ResultItem
        v-for="r in results"
        :key="r.chunk_id"
        :result="r"
        :query="query"
      />
      <div
        v-if="!results.length && query"
        class="rounded border border-slate-200 bg-white p-4 text-sm text-slate-500"
      >
        {{ message ?? '검색 결과가 없습니다.' }}
      </div>
    </template>
  </section>
</template>
