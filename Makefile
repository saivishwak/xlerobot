.PHONY: setup teleop pi05-server pi05-infer webapp webapp-build webapp-dev webapp-frontend webapp-backend clean help

TASK         ?= Pick up the object and place it in the bin
EPISODES     ?= 5
EPISODE_TIME ?= 60
WEBAPP_PORT  ?= 5000
WEBAPP_HOST  ?= 0.0.0.0

help:
	@echo "Targets:"
	@echo "  setup           One-time install (lerobot + XLeVR + openpi-client)"
	@echo "  webapp          Build frontend + run Flask backend on :$(WEBAPP_PORT)"
	@echo "  webapp-build    Build the React frontend (output → webapp/backend/static)"
	@echo "  webapp-dev      Run Flask (:5000) + Vite (:5173) for hot-reload development"
	@echo "  teleop          VR teleop + LeRobot dataset capture"
	@echo "  pi05-server     Start the openpi pi0.5 WebSocket server"
	@echo "  pi05-infer      Run the pi0.5 inference loop on the robot"
	@echo ""
	@echo "Overrides: TASK='...'  EPISODES=N  EPISODE_TIME=SEC  WEBAPP_PORT=5000"

setup:
	bash scripts/setup_xlerobot.sh

webapp-build:
	cd webapp/frontend && pnpm install --frozen-lockfile && pnpm build

webapp: webapp-build
	uv run flask --app webapp.backend.app:create_app run --host $(WEBAPP_HOST) --port $(WEBAPP_PORT)

webapp-backend:
	uv run flask --app webapp.backend.app:create_app run --host $(WEBAPP_HOST) --port $(WEBAPP_PORT) --debug

webapp-frontend:
	cd webapp/frontend && pnpm dev

webapp-dev:
	@echo "Run these in two terminals (Vite proxies /api → Flask):"
	@echo "  make webapp-backend"
	@echo "  make webapp-frontend"

teleop:
	uv run python scripts/run_vr_teleop_capture.py \
		--task "$(TASK)" \
		--episodes $(EPISODES) \
		--episode-time $(EPISODE_TIME)

pi05-server:
	bash scripts/run_openpi_server.sh

pi05-infer:
	uv run python scripts/run_pi05_inference.py \
		--task "$(TASK)" \
		--episodes $(EPISODES) \
		--episode-time $(EPISODE_TIME)

clean:
	@echo "Removing project .venv (lerobot/openpi venvs unchanged)..."
	rm -rf .venv uv.lock
