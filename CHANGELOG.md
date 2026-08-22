# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · versioning is by date (research project).

## [Unreleased]

### Added
- Evaluation toolkit: `scorer/` (WER, DNSMOS, NISQA, tool-call, LLM judge) behind a
  registry, `evaluation/` pipeline (QuestionSet → generation → scoring → JSON report),
  `lfm2-evaluate` CLI. Missing optional dependencies degrade the report instead of
  failing the campaign.
- RunPod serverless inference: `lfm2_audio.remote.LiquidAudioClient` (blocking +
  streaming), pydantic wire protocol shared with `infra/handler.py`, two worker
  images (`Dockerfile.serve` vLLM-Omni, `Dockerfile.serve.liquid` reference backend).
- SkyPilot training path: `infra/sky_train.yaml` + idempotent `infra/prepare_data.py`.
- Gradio test app (`make app`) with a session cost meter (default budget: $1).
- Makefile: sectioned help, `init` (devcontainer), `check` (lint + types + tests +
  secret guard), `ready`, `sanitize`, RunPod/SkyPilot targets.

### Changed
- CLI package reorganized by domain: `cli/data/`, `cli/train/`, `cli/eval/`,
  `cli/serve/`; command names (`lfm2-*`) are unchanged.
- Vague module names clarified: `ds/config.py` → `ds/inference_config.py`,
  `core/lazy.py` → `core/lazy_component.py`, `remote/codec.py` → `remote/wav_base64.py`,
  `vllm_plugin/lfm2_audio*.py` → `omni_model.py` / `stage_ar.py` / `stage_code2wav.py`.
- Network payloads are validated with pydantic at the boundary (no raw dict access).

### Added (weekend plan)
- EN baseline evaluation kit: `benchmark/baseline_en/questions.jsonl` (24 spoken-EN
  questions), `notebooks/colab_baseline_eval.ipynb` (L4) running WER/DNSMOS/NISQA
  and tool-calling campaigns via `lfm2-evaluate` (steps 1-2).
- `lfm2-dataset-inventory` CLI + `data_prep/hub_inventory.py`: Markdown inventory
  of a Hub author's datasets (step 3).
- `docs/weekend_plan.md`: 7-step plan, status and comparability rules.

### Conventions (from 2026-08-22)
- New and modified code comments are written in English.
- Work branches are named `rd/pr_rca_{action}` (action ≤ 2 words).
- Every push updates this changelog; pre-commit must pass on a clean test run.
