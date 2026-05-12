import { chromium } from 'playwright'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const PDF = path.join(ROOT, 'backend/data/pdfs/NAVER_business_report_2022.pdf')
const OUT = '/tmp/sps_e2e'
const BASE = process.env.E2E_BASE || 'http://localhost:8000'

const log = (m) => console.log(`[e2e] ${m}`)

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()

page.on('pageerror', (e) => console.error('[pageerror]', e.message))
page.on('console', (msg) => {
  if (msg.type() === 'error') console.error('[console.error]', msg.text())
})

await page.goto(BASE, { waitUntil: 'networkidle' })
await page.screenshot({ path: `${OUT}_01_initial.png`, fullPage: true })
log(`01 loaded: title="${await page.title()}"`)

// 초기 상태 뱃지
const initialStatus = (await page.locator('[class*="rounded-full"]').first().textContent())?.trim()
log(`initial badge: ${initialStatus}`)

// 업로드 — hidden input 직접 setInputFiles
const uploadInput = page.locator('input[type=file][multiple]')
await uploadInput.setInputFiles(PDF)
log('02 file selected, waiting for indexing...')

// 업로드 → 인덱싱 완료까지 (성공 메시지 출현)
await page.waitForSelector('text=/uploaded and indexed successfully|업로드 오류/', {
  timeout: 10 * 60 * 1000,
})
await page.screenshot({ path: `${OUT}_02_uploaded.png`, fullPage: true })
const uploadResult = (await page.locator('text=/uploaded and indexed successfully|업로드 오류/').textContent())?.trim()
log(`upload result: ${uploadResult}`)

// status badge가 ready로 갱신될 때까지 대기 (UploadZone이 emit, HomeView가 refresh 호출)
await page.waitForSelector('text=/준비 완료/', { timeout: 30_000 })
const readyStatus = (await page.locator('[class*="rounded-full"]').first().textContent())?.trim()
log(`status updated: ${readyStatus}`)

// 검색
await page.locator('input[type=search]').fill('네이버 광고 사업')
await page.locator('input[type=search]').press('Enter')
await page.waitForSelector('article', { timeout: 60_000 })
const cards = await page.locator('article').all()
log(`03 results: ${cards.length} cards`)
const firstCard = await page.locator('article').first().textContent()
log(`  top card excerpt: ${firstCard?.slice(0, 120).replace(/\s+/g, ' ')}...`)
await page.screenshot({ path: `${OUT}_03_search.png`, fullPage: true })

// 단일 파일 검색 토글
await page.locator('input[type=checkbox]').check()
await page.locator('input[type=file]:not([multiple])').setInputFiles(PDF)
await page.locator('input[type=search]').fill('데이터 센터')
await page.locator('input[type=search]').press('Enter')
// 단일 파일 검색은 업로드 + 임베딩이 즉시 일어남
await page.waitForSelector('article', { timeout: 5 * 60 * 1000 })
const filecards = await page.locator('article').all()
log(`04 file-mode results: ${filecards.length} cards`)
await page.screenshot({ path: `${OUT}_04_filemode.png`, fullPage: true })

await browser.close()
log('DONE')
