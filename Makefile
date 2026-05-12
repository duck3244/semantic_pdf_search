PYTHON ?= /home/duck/miniconda3/envs/py310_pt/bin/python
NPM ?= npm

BACKEND_DIR := backend
FRONTEND_DIR := frontend

.PHONY: help install backend-dev frontend-dev dev build clean

help:
	@echo "Targets:"
	@echo "  install        프론트 의존성 설치 (npm ci)"
	@echo "  backend-dev    백엔드 dev 서버 (RELOAD=1, port 8000)"
	@echo "  frontend-dev   프론트 dev 서버 (Vite, port 5173, /api -> :8000)"
	@echo "  dev            backend-dev와 frontend-dev 동시 실행 (Ctrl+C로 둘 다 종료)"
	@echo "  build          프론트 운영 빌드 (frontend/dist 생성 -> 백엔드가 자동 마운트)"
	@echo "  clean          빌드/캐시 산출물 제거"

install:
	cd $(FRONTEND_DIR) && $(NPM) ci

backend-dev:
	cd $(BACKEND_DIR) && RELOAD=1 $(PYTHON) main.py

frontend-dev:
	cd $(FRONTEND_DIR) && $(NPM) run dev

# 두 dev 서버를 동시에 실행. 한쪽이 죽으면 다른 쪽도 종료.
dev:
	@trap 'kill 0' EXIT INT TERM; \
	$(MAKE) -s backend-dev & \
	$(MAKE) -s frontend-dev & \
	wait

build:
	cd $(FRONTEND_DIR) && $(NPM) run build

clean:
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite
	find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} +
