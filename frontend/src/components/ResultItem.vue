<script setup lang="ts">
import { computed } from 'vue'
import type { SearchResult } from '@/types/api'

const props = defineProps<{ result: SearchResult; query: string }>()

// 정규식 메타문자 이스케이프 + 토큰 분리 (XSS 방지: textContent만 사용)
function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

interface Segment {
  text: string
  highlight: boolean
}

const segments = computed<Segment[]>(() => {
  const text = props.result.text ?? ''
  const terms = props.query
    .split(/\s+/)
    .filter((t) => t.length > 1)
    .map(escapeRegex)
  if (terms.length === 0) return [{ text, highlight: false }]
  const re = new RegExp(`(${terms.join('|')})`, 'gi')
  const parts = text.split(re)
  return parts
    .filter((p) => p !== '')
    .map((p, i) => ({ text: p, highlight: i % 2 === 1 }))
})

const percent = computed(() => (props.result.similarity * 100).toFixed(1))
</script>

<template>
  <article class="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
    <header class="mb-2 flex flex-wrap items-baseline justify-between gap-2">
      <div class="min-w-0">
        <div class="truncate font-medium text-slate-900" :title="result.filename">
          {{ result.filename }}
        </div>
        <div class="text-xs text-slate-500">페이지 {{ result.page }} · 단락 {{ result.paragraph }}</div>
      </div>
      <span class="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
        유사도 {{ percent }}%
      </span>
    </header>
    <p class="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700">
      <template v-for="(seg, i) in segments" :key="i">
        <mark v-if="seg.highlight" class="rounded bg-yellow-200 px-0.5">{{ seg.text }}</mark>
        <template v-else>{{ seg.text }}</template>
      </template>
    </p>
  </article>
</template>
