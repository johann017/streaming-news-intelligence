.PHONY: setup test lint run-pipeline clean-data deploy-firestore-rules help

# ── Setup ──────────────────────────────────────────────────────────────────

setup:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Copying .env.example → .env (edit with your values)"
	@[ -f .env ] || cp .env.example .env

# ── Testing ────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

# ── Linting ────────────────────────────────────────────────────────────────

lint:
	python -m ruff check shared/ services/ backend/ tests/ || true
	python -m mypy shared/ services/ backend/ --ignore-missing-imports --no-error-summary || true

# ── Local pipeline run ─────────────────────────────────────────────────────

run-pipeline:
	@echo "Running full pipeline locally..."
	@[ -f .env ] && export $$(cat .env | grep -v '^#' | xargs) ; python backend/workers/pipeline.py

run-ingestion:
	python services/ingestion/run.py

run-processing:
	python services/processing/run.py

run-clustering:
	python services/clustering/run.py

run-ranking:
	python services/ranking/run.py

run-summarization:
	python services/summarization/run.py

# ── Firebase ───────────────────────────────────────────────────────────────

deploy-firestore-rules:
	@echo "Deploying Firestore rules and indexes..."
	@[ -n "$$FIREBASE_PROJECT_ID" ] || (echo "ERROR: FIREBASE_PROJECT_ID not set" && exit 1)
	bash infra/firebase-deploy.sh

cleanup-firestore:
	@echo "Running manual Firestore cleanup..."
	python backend/workers/cleanup.py

# ── Maintenance ────────────────────────────────────────────────────────────

clean-data:
	@echo "Wiping local data/ directory (pipeline state)..."
	@rm -f data/raw_articles.json data/normalized_articles.json \
	       data/clusters.json data/ranked_clusters.json data/events.json \
	       data/cursors.json data/seen_ids.json data/notified_ids.json \
	       data/embedding_cache.pkl data/pipeline_run_log.json
	@echo "Done."

# ── Help ───────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  make setup                  Install deps, copy .env.example"
	@echo "  make test                   Run all tests"
	@echo "  make test-cov               Run tests with coverage report"
	@echo "  make lint                   Run ruff + mypy"
	@echo "  make run-pipeline           Run full pipeline locally"
	@echo "  make run-ingestion          Run ingestion stage only"
	@echo "  make run-processing         Run processing stage only"
	@echo "  make run-clustering         Run clustering stage only"
	@echo "  make run-ranking            Run ranking stage only"
	@echo "  make run-summarization      Run summarization stage only"
	@echo "  make deploy-firestore-rules Deploy Firestore rules/indexes"
	@echo "  make cleanup-firestore      Delete expired/excess Firestore docs"
	@echo "  make clean-data             Wipe local pipeline state files"
