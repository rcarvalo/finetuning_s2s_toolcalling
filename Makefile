# Tâches du projet. Tout passe par `uv` — jamais pip/conda.
#
#   make            → l'aide
#   make init       → ouvre le devcontainer
#   make check      → ce que la CI vérifie (lint + types + tests + secrets)
#   make sanitize   → supprime tout ce qui est régénérable
#   make app        → app Gradio de test de l'endpoint serverless

.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory
SHELL := /bin/bash

# Arguments libres : `make test -k waveform` → RUN_ARGS = "-k waveform".
RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(RUN_ARGS):;@:)

# printf '%b' interprète \033 partout (le `echo` de bash ne le fait pas sans -e).
ECHO := printf '%b\n'

_RED=\033[31m
_GREEN=\033[32m
_YELLOW=\033[33m
_CYAN=\033[36m
_BOLD=\033[1m
_END=\033[0m

UV := uv
WORKSPACE := lfm2-audio
# Dossiers lourds NON régénérables : `sanitize` les signale, ne les supprime jamais.
ARTIFACT_DIRS := outputs exports checkpoints wandb data/audio_tc_en

##@ Aide

help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n$(_BOLD)%s$(_END)\n", substr($$0, 5) } \
		/^[a-zA-Z_][a-zA-Z0-9_-]*:.*##/ { printf "  $(_CYAN)%-20s$(_END) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@$(ECHO) ""

##@ Environnement

init: _require-docker ## Ouvre le devcontainer (env reproductible, hooks inclus)
	@if command -v devcontainer >/dev/null 2>&1; then \
		$(ECHO) "$(_CYAN)Démarrage du devcontainer…$(_END)"; \
		devcontainer up --workspace-folder .; \
		$(ECHO) "$(_GREEN)Prêt$(_END) — attacher VS Code : Dev Containers: Attach to Running Container"; \
	else \
		$(ECHO) "$(_YELLOW)CLI devcontainer absente.$(_END) Deux options :"; \
		$(ECHO) "  • VS Code : F1 → $(_BOLD)Dev Containers: Reopen in Container$(_END)"; \
		$(ECHO) "  • CLI     : npm i -g @devcontainers/cli && make init"; \
		exit 1; \
	fi

install: _require-uv ## Crée .venv et installe le projet + le groupe dev
	$(UV) sync

install-serving: _require-uv ## + extra serving (vLLM-Omni, GPU uniquement)
	$(UV) sync --extra serving

install-app: _require-uv ## + extra app (Gradio, sans GPU)
	$(UV) sync --extra app

install-all: _require-uv ## + tous les extras (GPU requis)
	$(UV) sync --all-extras

hooks: _require-uv ## Installe les hooks pre-commit
	$(UV) run pre-commit install

lock: _require-uv ## Rafraîchit uv.lock
	$(UV) lock

##@ Qualité

format: ## Formate le code (ruff format)
	$(UV) run ruff format .

lint: ## Lint + imports (ruff check --fix)
	$(UV) run ruff check --fix .

lint-check: ## Lint sans écrire (CI)
	$(UV) run ruff format --check .
	$(UV) run ruff check .

typecheck: ## mypy --strict sur le paquet
	$(UV) run mypy

test: ## Tests sans GPU
	$(UV) run pytest -q -m "not gpu" $(RUN_ARGS)

test-cov: ## Tests + couverture
	$(UV) run pytest -q -m "not gpu" --cov --cov-report=term-missing

test-gpu: ## Tests GPU (checkpoints requis, cf. docs/)
	$(UV) run pytest -q -m gpu

preco: ## Lance pre-commit sur tous les fichiers
	$(UV) run pre-commit run --all-files $(RUN_ARGS)

check-secrets: ## Vérifie qu'aucun secret ne partirait dans un push
	@fail=0; \
	if git ls-files --error-unmatch .env >/dev/null 2>&1; then \
		$(ECHO) "  $(_RED)✗ .env est suivi par git$(_END) — git rm --cached .env"; fail=1; \
	fi; \
	if git grep -nIE --cached '(rpa_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|tvly-[A-Za-z0-9]{20,})' \
		-- . ':!uv.lock' ':!*.ipynb'; then \
		$(ECHO) "  $(_RED)✗ secret détecté dans le contenu indexé$(_END) — le retirer AVANT de committer"; fail=1; \
	fi; \
	test $$fail -eq 0 && $(ECHO) "  $(_GREEN)✓ aucun secret dans l'index$(_END)"; \
	exit $$fail

check: lint-check typecheck test check-secrets ## Tout ce que la CI vérifie + garde secrets
	@$(ECHO) "$(_GREEN)✓ check complet$(_END)"

ready: check ## check + résumé de ce qui partirait dans un push
	@$(ECHO) "\n$(_BOLD)Branche$(_END)  $$(git branch --show-current) → $$(git remote get-url origin 2>/dev/null || echo 'aucun remote')"
	@$(ECHO) "$(_BOLD)Indexé$(_END)   $$(git diff --cached --numstat | wc -l | tr -d ' ') fichier(s)"
	@$(ECHO) "$(_BOLD)Non suivi$(_END) $$(git ls-files --others --exclude-standard | wc -l | tr -d ' ') fichier(s) — git add si voulus"
	@$(ECHO) "\n$(_BOLD)Plus gros fichiers indexés$(_END)"
	@git diff --cached --name-only --diff-filter=ACM | xargs -I{} du -k {} 2>/dev/null | sort -rn | head -5 | awk '{printf "  %6s Ko  %s\n", $$1, $$2}'
	@$(ECHO) "\n$(_GREEN)Prêt à pousser$(_END) : git push -u origin $$(git branch --show-current)"

##@ Usage

app: _require-endpoint ## App Gradio de test de l'endpoint serverless (budget 1 $)
	@set -a; . ./.env; set +a; \
	$(UV) run --extra app python app/gradio_app.py

demo: ## Démo S2S (CKPT=<chemin ou repo HF>)
	$(UV) run lfm2-demo --checkpoint "$(CKPT)" $(RUN_ARGS)

bench: ## Bench TTFA (CKPT=<chemin ou repo HF>)
	$(UV) run lfm2-bench --checkpoint "$(CKPT)" $(RUN_ARGS)

smoke: ## Smoke test du plugin vLLM-Omni (CKPT=<chemin>)
	$(UV) run lfm2-smoke --checkpoint "$(CKPT)" $(RUN_ARGS)

##@ Infra RunPod

serve-build: _require-docker ## Build local de l'image serverless liquid (linux/amd64)
	docker build --platform linux/amd64 -f infra/Dockerfile.serve.liquid \
		-t lfm2-audio-serverless:liquid .

train: _require-sky ## Lance un batch d'entraînement sur RunPod (pod détruit à la fin)
	@set -a; . ./.env; set +a; \
	sky launch -c liquid-train infra/sky_train.yaml --down \
		--env HF_TOKEN --env WANDB_API_KEY $(RUN_ARGS)

train-exec: _require-sky ## Relance un batch sur le cluster déjà up
	@set -a; . ./.env; set +a; \
	sky exec liquid-train infra/sky_train.yaml --env HF_TOKEN --env WANDB_API_KEY $(RUN_ARGS)

train-logs: _require-sky ## Suit les logs du batch en cours
	sky logs liquid-train

train-down: _require-sky ## Détruit le cluster d'entraînement
	sky down liquid-train

##@ Nettoyage

clean: ## Supprime les caches et artefacts de build (régénérables)
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build site *.egg-info
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '.DS_Store' -delete 2>/dev/null || true
	@$(ECHO) "$(_GREEN)✓ caches supprimés$(_END)"

sanitize: clean ## clean + .venv + fichiers ignorés ; signale les artefacts lourds (FORCE=1 : sans confirmation)
	@if [ "$(FORCE)" != "1" ]; then \
		read -r -p "Supprimer .venv et TOUS les fichiers ignorés par git (hors artefacts lourds) ? [y/N] " a; \
		[ "$$a" = "y" ] || { $(ECHO) "$(_YELLOW)annulé$(_END)"; exit 1; }; \
	fi
	@rm -rf .venv
	@git clean -Xdf $(addprefix -e ,$(ARTIFACT_DIRS)) >/dev/null 2>&1 || true
	@$(ECHO) "$(_GREEN)✓ environnement remis à neuf$(_END) — relancer : make install"
	@$(ECHO) "\n$(_BOLD)Artefacts lourds conservés$(_END) (non régénérables — supprimer à la main si besoin) :"
	@for d in $(ARTIFACT_DIRS); do \
		[ -d "$$d" ] && $(ECHO) "  $(_YELLOW)$$(du -sh $$d 2>/dev/null | cut -f1)$(_END)\t$$d"; \
	done; true

##@ Gardes internes

_require-uv:
	@command -v uv >/dev/null 2>&1 || { \
		$(ECHO) "$(_RED)uv absent$(_END) — curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

_require-docker:
	@command -v docker >/dev/null 2>&1 || { $(ECHO) "$(_RED)docker absent$(_END)"; exit 1; }
	@docker info >/dev/null 2>&1 || { \
		$(ECHO) "$(_RED)démon Docker arrêté$(_END) — lancer Docker Desktop puis réessayer"; exit 1; }

_require-sky:
	@command -v sky >/dev/null 2>&1 || { \
		$(ECHO) "$(_RED)SkyPilot absent$(_END) — uv tool install 'skypilot[runpod]' && runpod config && sky check runpod"; exit 1; }
	@test -f .env || { $(ECHO) "$(_RED).env manquant$(_END) — cp .env.example .env puis remplir HF_TOKEN"; exit 1; }

_require-endpoint:
	@test -f .env || { $(ECHO) "$(_RED).env manquant$(_END) — cp .env.example .env"; exit 1; }
	@grep -qE '^RUNPOD_API_KEY=.+' .env || { \
		$(ECHO) "$(_RED)RUNPOD_API_KEY vide dans .env$(_END)"; exit 1; }
	@grep -qE '^RUNPOD_ENDPOINT_ID=.+' .env || { \
		$(ECHO) "$(_RED)RUNPOD_ENDPOINT_ID vide dans .env$(_END) — créer l'endpoint d'abord (infra/README.md §1)"; exit 1; }

.PHONY: help init install install-serving install-app install-all hooks lock \
	format lint lint-check typecheck test test-cov test-gpu preco check-secrets check ready \
	app demo bench smoke serve-build train train-exec train-logs train-down \
	clean sanitize _require-uv _require-docker _require-sky _require-endpoint
