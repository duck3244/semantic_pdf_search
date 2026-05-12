import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'

// hash mode: 서버 라우팅 없이 새로고침 안전, FastAPI StaticFiles 마운트와 충돌 X
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [{ path: '/', name: 'home', component: HomeView }],
})
