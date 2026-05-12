<script setup lang="ts">
import { ref } from 'vue'
import StatusBadge from '@/components/StatusBadge.vue'
import UploadZone from '@/components/UploadZone.vue'
import SearchPanel from '@/components/SearchPanel.vue'

const statusRef = ref<InstanceType<typeof StatusBadge> | null>(null)

function onIndexed() {
  statusRef.value?.refresh()
}
</script>

<template>
  <div class="mx-auto flex max-w-5xl flex-col gap-6 p-6">
    <header class="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 class="text-2xl font-bold text-slate-900">PDF 시맨틱 검색</h1>
        <p class="mt-1 text-sm text-slate-500">
          ko-sroberta-multitask + FAISS 기반 한국어 의미 검색 (CPU 전용)
        </p>
      </div>
      <StatusBadge ref="statusRef" />
    </header>

    <UploadZone @indexed="onIndexed" />
    <SearchPanel />
  </div>
</template>
