<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getStatus } from '@/api/search'
import { describeError } from '@/api/client'
import type { StatusResponse } from '@/types/api'

const status = ref<StatusResponse | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)

async function refresh() {
  loading.value = true
  error.value = null
  try {
    status.value = await getStatus()
  } catch (e) {
    error.value = describeError(e)
  } finally {
    loading.value = false
  }
}

const tone = computed(() => {
  if (error.value) return 'bg-red-100 text-red-700 border-red-200'
  if (status.value?.status === 'ready') return 'bg-emerald-100 text-emerald-700 border-emerald-200'
  return 'bg-amber-100 text-amber-700 border-amber-200'
})

onMounted(refresh)
defineExpose({ refresh })
</script>

<template>
  <div :class="['inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm', tone]">
    <span class="inline-block h-2 w-2 rounded-full bg-current opacity-70" />
    <template v-if="loading">상태 확인 중…</template>
    <template v-else-if="error">백엔드 오류: {{ error }}</template>
    <template v-else-if="status?.status === 'ready'">
      준비 완료 · 문서 {{ status.indexed_documents }} / 청크 {{ status.indexed_chunks.toLocaleString() }} ·
      {{ status.vector_dimension }}d
    </template>
    <template v-else>
      인덱스 비어 있음 — PDF를 업로드해 주세요
    </template>
    <button
      class="ml-1 rounded px-1.5 text-xs underline opacity-70 hover:opacity-100"
      @click="refresh"
      :disabled="loading"
    >
      새로고침
    </button>
  </div>
</template>
