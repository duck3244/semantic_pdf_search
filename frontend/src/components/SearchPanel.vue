<script setup lang="ts">
import { ref } from 'vue'
import { useSearch } from '@/composables/useSearch'
import ResultList from './ResultList.vue'

const query = ref('')
const topK = ref(5)
const fileMode = ref(false)
const file = ref<File | null>(null)
const isComposing = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const { loading, results, message, errorMessage, lastQuery, run } = useSearch()

function onSubmit() {
  if (isComposing.value) return // IME 조합 중 Enter 무시
  if (fileMode.value && !file.value) return
  run(query.value, topK.value, fileMode.value ? file.value : null)
}

function pickFile() {
  fileInput.value?.click()
}

function onFilePicked(e: Event) {
  const input = e.target as HTMLInputElement
  file.value = input.files?.[0] ?? null
}

function clearFile() {
  file.value = null
  if (fileInput.value) fileInput.value.value = ''
}
</script>

<template>
  <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-base font-semibold text-slate-900">검색</h2>
      <label class="flex items-center gap-2 text-xs text-slate-600">
        <input
          type="checkbox"
          v-model="fileMode"
          class="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
        />
        업로드하지 않고 단일 PDF에서만 검색
      </label>
    </div>

    <div class="flex flex-col gap-3 md:flex-row md:items-center">
      <input
        v-model="query"
        type="search"
        placeholder="예: 네이버 광고 사업 / FAISS 벡터 검색"
        class="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-400 focus:outline-none focus:ring focus:ring-blue-100"
        @compositionstart="isComposing = true"
        @compositionend="isComposing = false"
        @keydown.enter.prevent="onSubmit"
        :disabled="loading"
      />
      <div class="flex items-center gap-2 text-sm text-slate-600">
        <label for="topk" class="whitespace-nowrap">결과 개수</label>
        <input
          id="topk"
          v-model.number="topK"
          type="number"
          min="1"
          max="50"
          class="w-16 rounded-md border border-slate-300 px-2 py-1 text-sm"
        />
      </div>
      <button
        type="button"
        class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="loading || !query.trim() || (fileMode && !file)"
        @click="onSubmit"
      >
        검색
      </button>
    </div>

    <div v-if="fileMode" class="mt-3 flex items-center gap-2 text-xs text-slate-600">
      <button
        type="button"
        class="rounded border border-slate-300 px-2 py-1 hover:bg-slate-50"
        @click="pickFile"
      >
        PDF 선택
      </button>
      <span v-if="file" class="truncate" :title="file.name">{{ file.name }}</span>
      <span v-else class="text-slate-400">선택된 파일 없음</span>
      <button
        v-if="file"
        type="button"
        class="text-slate-500 underline"
        @click="clearFile"
      >
        제거
      </button>
      <input
        ref="fileInput"
        type="file"
        accept=".pdf,application/pdf"
        class="hidden"
        @change="onFilePicked"
      />
    </div>

    <ResultList
      :results="results"
      :query="lastQuery"
      :loading="loading"
      :message="message"
      :error-message="errorMessage"
    />
  </section>
</template>
