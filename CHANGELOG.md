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
- `docs/colab.md`: verified remote-execution procedure via `google-colab-cli`
  (`colab new --gpu L4` + `colab exec -f`), which the MCP browser bridge cannot do
  (it has no runtime-management tool). Confirmed on an L4 (23034 MiB, Python 3.13.15)
  that vLLM 0.22.x and vllm-omni 0.22.0 still resolve — the versions the plugin targets.

### Fixed
- `serving/backends/liquid.py`: hand liquid-audio a `Path`, not a `str`.
  `get_model_dir` overloads its argument by type (str = Hub repo id via
  snapshot_download, Path = local directory), so every locally materialized
  checkpoint failed with HFValidationError. Found by the first baseline run on
  an L4. LoRA helpers are now imported inside `_merge_adapter`: peft and
  safetensors are training-only extras that broke a plain serving install.

### Added (step 7 tooling)
- `evaluation/comparison.py` + `lfm2-eval-compare`: baseline vs candidate table,
  metric direction read from the report, comparability warnings, exit code 1 on
  regression so a training loop can gate on it.

### Results
- EN baselines measured on Colab L4 (`docs/baseline_en.md`, reports in `reports/`):
  audio WER 0.086 / DNSMOS OVRL 3.329 (NISQA unavailable, no weights);
  tool calling 0.333 — the vanilla model emits **zero** tool calls on the
  8 positive cases while abstaining correctly on all 4 negatives.

### Added (steps 3-5)
- `lfm2-dataset-inventory` run on `Rcarvalo`: 50 repos catalogued
  (`docs/dataset_inventory.md`). Selection and gaps in `docs/dataset_selection.md`.
- `data_prep/curation.py` + `lfm2-dataset-curate`: merge dialogue sources,
  deduplicate on the normalized user utterance (contractions folded) and refuse
  anything present in the held-out split; exit 1 on leakage. 15 tests.

### Conventions (from 2026-08-22)
- New and modified code comments are written in English.
- Work branches are named `rd/pr_rca_{action}` (action ≤ 2 words).
- Every push updates this changelog; pre-commit must pass on a clean test run.
