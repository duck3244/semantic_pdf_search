<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useUpload } from '@/composables/useUpload'

const emit = defineEmits<{
  (e: 'indexed'): void
}>()

const { phase, percent, lastResult, errorMessage, validate, upload, reset } = useUpload()

const dragOver = ref(false)
const rejected = ref<string[]>([])
const fileInput = ref<HTMLInputElement | null>(null)

const busy = computed(() => phase.value === 'uploading' || phase.value === 'indexing')

const phaseLabel = computed(() => {
  switch (phase.value) {
    case 'uploading':
      return `업로드 중 ${percent.value}%`
    case 'indexing':
      return '서버에서 인덱싱 중… (최대 수 분 소요)'
    case 'done':
      return '완료'
    case 'error':
      return '실패'
    default:
      return ''
  }
})

async function handle(files: FileList | File[]) {
  const list = Array.from(files)
  const { ok, rejected: rej } = validate(list)
  rejected.value = rej
  if (ok.length === 0) return
  await upload(ok)
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  if (e.dataTransfer?.files?.length) handle(e.dataTransfer.files)
}

function pickFiles() {
  fileInput.value?.click()
}

function onPicked(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.length) handle(input.files)
  input.value = ''
}

watch(phase, (p) => {
  if (p === 'done') emit('indexed')
})
</script>

<template>
  <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-base font-semibold text-slate-900">PDF 업로드 & 인덱싱</h2>
      <span class="text-xs text-slate-400">단일 파일 최대 100MB · 한국어 PDF 권장</span>
    </div>

    <div
      :class="[
        'flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition',
        dragOver ? 'border-blue-400 bg-blue-50' : 'border-slate-200 bg-slate-50',
        busy ? 'pointer-events-none opacity-60' : 'cursor-pointer hover:bg-slate-100',
      ]"
      @dragover.prevent="dragOver = true"
      @dragleave.prevent="dragOver = false"
      @drop.prevent="onDrop"
      @click="pickFiles"
    >
      <p class="text-sm text-slate-700">
        여기로 PDF를 드래그하거나 <span class="font-semibold text-blue-600">클릭해서 선택</span>하세요
      </p>
      <p class="mt-1 text-xs text-slate-400">여러 개 동시 업로드 가능</p>
      <input
        ref="fileInput"
        type="file"
        accept=".pdf,application/pdf"
        multiple
        class="hidden"
        @change="onPicked"
      />
    </div>

    <div v-if="busy || phase === 'done' || phase === 'error'" class="mt-4 space-y-2">
      <div class="text-sm text-slate-700">{{ phaseLabel }}</div>
      <div v-if="phase === 'uploading'" class="h-2 w-full overflow-hidden rounded bg-slate-100">
        <div class="h-full bg-blue-500 transition-all" :style="{ width: percent + '%' }" />
      </div>
      <div v-if="phase === 'indexing'" class="h-2 w-full overflow-hidden rounded bg-slate-100">
        <div class="h-full w-1/3 animate-pulse bg-amber-400" />
      </div>
    </div>

    <div v-if="rejected.length" class="mt-3 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
      거부됨: {{ rejected.join(', ') }}
    </div>

    <div v-if="errorMessage" class="mt-3 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
      업로드 오류: {{ errorMessage }}
    </div>

    <div v-if="phase === 'done' && lastResult" class="mt-3 rounded border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-800">
      <p class="font-medium">{{ lastResult.message }}</p>
      <p v-if="lastResult.indexed_files.length" class="mt-1">
        인덱싱:
        <span v-for="f in lastResult.indexed_files" :key="f" class="ml-1 break-all">· {{ f }}</span>
      </p>
      <p v-if="lastResult.skipped_files.length" class="mt-1">
        건너뜀: {{ lastResult.skipped_files.join(', ') }}
      </p>
      <button class="mt-2 text-emerald-700 underline" @click="reset">닫기</button>
    </div>
  </section>
</template>
