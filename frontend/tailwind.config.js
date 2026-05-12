/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,ts,js}'],
  safelist: [
    // 검색 결과 하이라이트용 — 빌드 시 purge 방지
    'bg-yellow-200',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'Pretendard',
          '-apple-system',
          'BlinkMacSystemFont',
          'system-ui',
          'Roboto',
          'Helvetica Neue',
          'Apple SD Gothic Neo',
          'Noto Sans KR',
          'Arial',
          'sans-serif',
        ],
      },
    },
  },
  plugins: [],
}
