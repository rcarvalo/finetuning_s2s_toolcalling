# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · versioning is by date (research project).

## [Unreleased]

### Added
- `lang_match` scorer — deterministic FR/EN mirroring check (function-word
  counting, no dependency, no API call): 1.0 when the reply's language matches
  `metadata.expected_lang` (falling back to `lang`), the metric the
  `lang_mirror` benchmark and the R3 ≥95 % gate read. Fails rather than
  guesses on signal-free replies. Off by default like `asr_wer`.
- FR source audit (phase 1.1) — `lfm2-fr-audit` samples each candidate FR
  source (`configs/audit/fr_sources.yaml`), measures the raw-audio ones with
  VERSA (DNSMOS/UTMOS/NISQA — the same metrics the gates read) plus a
  label-cleanliness WER (faster-whisper fr re-listening vs the shipped
  transcript), and writes the comparison to `docs/fr_data_audit.md`.
  Aggregation logic in `data_prep/fr_source_audit.py` (tested). Sources
  without raw audio are audited on metadata only: `pilot-125h` is packed
  tensors (and already pilot-validated), `emilia-yodas-fr` ships codec codes —
  decoding them would judge the codec, not the corpus. `dialogue-tts-1000h`
  is a wav+metadata.jsonl audiofolder, sampled with an even stride so one
  recording batch cannot masquerade as the whole corpus.
- `asr_wer` scorer — plain text-vs-text WER of the model's reply against the
  reference transcript, the D1 gate metric on `fleurs_fr_asr`. The audio `wer`
  scorer measures the TTS path (it re-transcribes generated speech); this one
  measures the listening path, shares the same Levenshtein and normalisation,
  and needs no transcriber. Registered but off by default: comparing a free
  reply to a transcript only makes sense on an ASR benchmark.
- FR benchmarks (phase 0A, task 6). `benchmark/fr_s2s/` (100 spoken-style FR
  questions), `benchmark/lang_mirror/` (20 FR / 20 EN / 20 code-switch cases
  with `meta.expected_lang` — the mirroring gate reads % of replies in the
  right language), `benchmark/fleurs_fr_asr/` (200 FLEURS-fr test clips) and
  `benchmark/cv_fr_asr/` (300 student-dataset clips, distillmos ≥ 3.5, ≤ 3
  clips/speaker). New `lfm2-asr-bench` CLI (selection logic in
  `data_prep/asr_bench.py`, tested) streams any HF audio dataset into this
  JSONL + 16 kHz WAV layout. Because the user's datasets and the student's
  overlap (both draw on Common Voice FR), each ASR benchmark ships
  `speakers.txt` AND `source_ids.txt`: the training mixer must hold these out
  of EVERY source, not just the one the benchmark was cut from.
- Bilingual eval plumbing (phase 0A of the FR/EN plan). ASR language now flows
  end to end: `ScoringConfig.asr_language` sets the campaign default, a sample
  carrying `metadata["lang"]` wins over it, and both Whisper backends accept the
  per-call override — a FR clip is no longer transcribed as English noise.
  `voice_eval` exposes `-T asr_language=fr`. The judge rubric gains
  `reasoning-v3` with a `language_match` criterion and the judge prompt states
  the expected language: answering EN to a FR question no longer scores 5/5.
  FR latency prompts (`PROMPTS_BY_LANGUAGE`, `lfm2-latency --lang fr`) mirror
  the EN set so the two TTFA series stay comparable.
- `evaluation/versa_runner.py` — bridge to the VERSA toolkit (isolated venv
  `versa-eval/`), the metric authority at gates: writes the `.scp` + YAML
  config, shells out to `scorer.py`, returns `key → {metric: value}`. Config
  presets for pseudo-MOS (DNSMOS+UTMOS), NISQA, Whisper WER and speaker
  similarity ship with it.
- Inspect AI is now the eval runner (extra `inspect`). Everything a campaign
  needs — parallelism, retries, several variants in one command, the log and its
  viewer — exists upstream and is better tested than a local rewrite. What is
  ours is what Inspect cannot supply: `inspect_bridge/provider.py` exposes
  LFM2.5-Audio as a model provider (`--model lfm2/<checkpoint> -M adapter=…`, or
  `--model lfm2-endpoint/<id>`) that hears a `ContentAudio` question and answers
  with the speech attached, so the viewer draws a player next to the score;
  `inspect_bridge/scorers.py` wraps our scorers instead of reimplementing them,
  so a number in the viewer is the number the report carries;
  `inspect_bridge/dataset.py` maps a question set, sending a spoken question as
  audio and never as its transcript.

      inspect eval python/lfm2_audio/inspect_bridge/task.py \
        --model lfm2/LiquidAI/LFM2.5-Audio-1.5B \
        -T scorers=tool_call,dnsmos,utmos --max-samples 1
      inspect view --log-dir logs

  `--max-samples 1` is load-bearing on a local checkpoint: the samples share one
  GPU, so raising it buys nothing and exhausts VRAM. Endpoints are the opposite.

### Removed
- `Campaign`, `CampaignConfig` and `lfm2-campaign`, written hours earlier: they
  duplicated `inspect eval` / `eval-set` (config-driven runs, bounded
  parallelism, per-variant logs) without their resume-after-failure or adaptive
  concurrency. A local layer shadowing a live tool inherits its bugs and none of
  its fixes.

### Added
- `ToolCallDiagnosis` (`evaluation/tool_call_diagnosis.py`): the anatomy of a
  tool-calling case, evidence kept. `score_case` computed which argument
  diverged and by how much, then discarded it one line later — a report could
  only say `call: false`. Every case now carries a single `outcome`
  (`correct_call`, `wrong_arguments`, `spurious_call`, `missing_call`,
  `wrong_tool`, `arity_mismatch`, `parse_error`, `unterminated_call`,
  `no_generation`, `correct_abstention`), the expected and predicted calls, the
  offending argument with its similarity and threshold, the parse errors and the
  raw span. `score_case` and `ToolCallScorer` delegate to it, so evaluation and
  training cannot drift apart. **Scores are unchanged** — only `details` grew.
- `evaluation/argument_match.py`: `diff_arguments` returns the reasons a call was
  rejected instead of a boolean; matching is now "no mismatch". `token_f1` and
  `ArgMatch` moved here and are re-exported from `toolcalling`.

### Fixed
- A tool-call span left open (no `<|tool_call_end|>`) counted as **no call at
  all**: a silent miss on a positive case, and a *false success* on a negative
  one, since abstaining was the correct answer there. vLLM strips the stop token
  and leaves spans open, so this was routine. Now labelled `unterminated_call`
  and counted as an attempted call.
- A positional argument (`web_search("query")`) is parsed as `_positional_0`,
  which no expected schema declares, so it read as a wrong *value* when the
  defect is a wrong *format*. Now reported as `positional_argument`.
- Two calls to the same tool made the name lists differ and read as
  `wrong_tool`, sending a reader after a routing bug that did not exist; that is
  now `arity_mismatch`.

### Changed
- `infra/colab_qwen_tts.py` batches Qwen3-TTS generation (`TTS_BATCH`, default
  8): the unitary loop measured ~4.5 wav/min on an L4 (~10 h for the Phase B
  assistant turns); batch 16 measures ~44/min (~1 h). Resumable as before.

### Fixed
- Serverless vLLM workers crash-looped at boot: FlashInfer's sampler JIT
  needs `nvcc` and the serve image ships no CUDA toolkit ("Could not find
  nvcc", StageEngineCoreProc dies during READY). Colab masked the issue —
  its VMs carry the full toolkit. `VLLM_USE_FLASHINFER_SAMPLER=0` is now
  baked into `Dockerfile.serve` (and set on the endpoint), falling back to
  vLLM's native sampler — no JIT, equivalent at batch size 1.

### Added
- Multi-turn conversation over the serverless endpoint: `TurnRequest` gains a
  `history` field (past `{role, text}` turns), the handler replays it into the
  model's conversation before generating, and `lfm2-voice` accumulates the
  session (assistant replies as text; past user audio cannot be replayed — the
  one-audio-per-conversation invariant — so user turns are empty markers).
  The session lives on the client: any worker can serve any turn, which is
  what keeps serverless scaling and FlashBoot free. Capped at the most recent
  exchange: replayed history is assistant-only, and stacking several one-sided
  turns makes the model blend unrelated topics (measured on the endpoint —
  after a French-word turn and a weather turn, "another one please" answered
  about weather while mentioning French). Lifting the cap needs the worker to
  return a transcript of the user's turn.
- `lfm2-voice` (extra `voice`): hands-free voice assistant over the serverless
  endpoint, ChatGPT-voice style — open mic, Silero VAD detects the end of the
  utterance, the reply audio plays as it streams out of the endpoint (first
  sound ≈ TTFA + one poll, instead of the bench app's wait-for-full-answer
  6-7 s). fastrtc WebRTC UI; turn logic in `bench/voice_turn.py`, tested
  without WebRTC. v1 endpoint is stateless → each utterance is single-turn.
- GitHub Actions workflow `build-serve-image`: builds `infra/Dockerfile.serve`
  and pushes it to `ghcr.io/rcarvalo/lfm2-serve-vllm` on every push to the
  serving branch (or manually via workflow_dispatch). Motivation: RunPod's
  GitHub builder is console-only and its queue stalled for 1h40+ with no way
  to observe or retry it; the endpoint now points at the GHCR image instead.
- `infra/handler.py`: the ~6.5 s first-generation warmup is now absorbed at
  worker boot (a tiny throwaway turn before `serverless.start`), so the first
  real job after a cold start sees steady-state TTFA (~0.3 s on vLLM) instead
  of paying the warmup. Best-effort — a warmup failure logs and never kills
  the worker.

### Fixed
- vLLM backend proven end-to-end on an L4: engine boots with the per-stage
  deploy config (piecewise CUDA graphs on stage 0), steady-state **TTFA 0.28 s,
  RTF 0.57** — the Colab-experience numbers. Two boot-killers found and fixed on
  the way: vLLM 0.22.1's AOT-compile default is incompatible with the torch it
  pins (`VLLM_USE_AOT_COMPILE=0` now baked into `Dockerfile.serve`), and the
  tool-calling demo's module-level `TORCHDYNAMO_DISABLE=1` kills any compiled
  stage (removed). First generation after boot pays ~6.5 s of warmup.
- The vLLM serverless image would have run in eager mode: `Dockerfile.serve`
  never copied `configs/`, and the package's deploy-config default only resolves
  inside a repo checkout — so the engine silently fell back to no CUDA graphs
  (~750 ms TTFA instead of ~300 ms). The image now ships
  `configs/serving/vllm_omni.yaml` and the handler honours `LFM2_DEPLOY_CONFIG`,
  failing loudly at boot if the file is missing. Caught before paying the build.
- `Dockerfile.serve` (vLLM worker) carried the same missing-C-compiler defect as
  the liquid one: `build-essential` added before it cost another 20-minute build.
- Serverless worker crashed at generation time with "Failed to find C compiler".
  `python:*-slim` ships none, and LFM2's causal convolution calls a Triton kernel
  that is JIT-compiled on first use — so `TORCHDYNAMO_DISABLE=1` does not help,
  the kernel is not reached through torch.compile. `Dockerfile.serve.liquid` now
  installs `build-essential`. The failure surfaces *after* the image is pulled,
  which reads as a slow cold start rather than a crash.
- DNSMOS scored every clip between 1.60 and 1.62, which read as "uniformly poor
  audio" and was in fact a broken metric. Three departures from
  microsoft/DNS-Challenge: the calibration used a cubic with an interior maximum
  (saturating at 1.62, so raw scores of 3.5-5.0 all collapsed into a 0.02-wide
  band) instead of the official quadratic polyfit; short clips were zero-padded
  to 9.01 s instead of tiled, feeding the model up to 45 % silence; windows did
  not overlap. Pinned by `tests/test_dnsmos_calibration.py`. The EN baseline's
  DNSMOS figures are void and must be recomputed.

### Added
- Listening bench (`lfm2-bench-app`): a Gradio app with two tabs — **Talk** for
  conversation, **Rate** to walk a test set scoring each answer 1-5 on
  intelligibility, naturalness and overall, plus a separate `derailed` flag for
  clips that loop or babble. Answers come from either a locally loaded checkpoint
  (`--checkpoint`, with a `liquid`/`vllm` backend selector) or a serverless
  endpoint (`--endpoint`), behind an `AnswerSource` protocol — so the UI runs on
  a laptop with no GPU. Verdicts append to `reports/human_ratings.jsonl` and the
  generated WAVs are kept, so a judgement can be revisited and compared across
  versions. A stateless endpoint makes the Talk tab single-turn, and the UI says so.
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

### Added (step 4b — trustworthy evaluation)
- `data_prep/splitting.py` + `lfm2-dataset-repack`: carve a stratified held-out
  split, deduplicated on the utterance so copies never straddle the boundary.
  Generic over a target reader, deterministic per seed. 11 tests.
- Curated dataset `Rcarvalo/tc-en-voice-agent-v1` (private): train 2729,
  `test_utterances` 200 (unseen wording, seen voices — statistical power),
  `test_voices` 12 (the original held-out voices — generalization). Tool
  distribution preserved within one point.

### Added (step 6 — training recipe)
- `TrainingConfig.num_epochs` + `training/step_budget.py`: a recipe can state
  its length in passes over the data; the step count is derived when the corpus
  size is known, and a loader without a length raises `TrainingConfigError`
  instead of a guessed budget. Kept out of `train_sft` so the policy stays
  testable without torch. 11 tests.
- `configs/training/tc_en_voice_agent_v1.yaml`: Phase A LoRA recipe on the
  curated corpus (3 epochs ≈ 1023 steps), encoder and audio heads frozen,
  adapter pushed every 250 steps (Colab reclaims sessions), tool_call scoring
  during training with the same objects as the final campaign.

### Added (pre-training review)
- `docs/pre_training_review.md`: LoRA-vs-full-fine-tune decision with an
  escalation ladder and a trigger per rung, the four blocking gaps found before
  launching, and the enrichment ranking.
- `infra/prepare_v1.py`: idempotent rehydrate → carve validation → pack, so a
  reclaimed Colab session costs minutes.

### Fixed (pre-training review)
- Training recipe scored nothing: `EvaluationScheduleConfig.enabled` defaults to
  `false` and `question_set` was empty, so the run would have trained blind for
  1023 steps. Now enabled on a validation slice, with `at_start` for a reference
  point, and the final test splits stay unwatched.

### Added (documentation)
- `docs/datasets.md`: the dataset reference — shared dialogue schema and its
  invariants, the three on-topic corpora with their real distributions, the
  curated splits and why the two test sets must not be merged, the adapters
  already on the Hub with their contamination caveat, the data-flow diagram and
  what is still missing.

### Fixed (evaluation measured the wrong thing)
- `lfm2-evaluate` never declared the tools. It built the model with
  DEFAULT_SYSTEM ("Respond with interleaved text and audio"), which names no
  tool, while training embeds the definitions in the system prompt. Every
  tool-calling campaign therefore asked the model to call tools it had never
  been told about — base model and fine-tune alike scored zero emissions. New
  `--tool-definitions` flag (`en` shortcut or a JSON path) renders the exact
  prompt training used; the choice is recorded in the report context, since two
  campaigns that declared different tools are not comparable. 4 tests.

### Fixed (v1 post-mortem — three inert or misleading mechanisms)
- Interleaved audio decode shredded tool-call spans during evaluation: the
  campaign now decodes text-only whenever no audio scorer is requested (the
  GenerationConfig docstring had warned about exactly this).
- In-training scoring silently skipped: the train CLI never passed a
  `generator_factory`, so `ScoringCallback` resolved to no generator. New
  `training/eval_generator.py` wraps the live model in the same serving stack
  as the final campaign (text-only, tools declared, eval-mode toggling).
- `evaluation.tool_definitions` on the schedule + one shared resolver
  (`evaluation/tool_prompt.py`) for the CLI and the callback.

### Added (v2)
- `configs/training/tc_en_voice_agent_v2.yaml`: each choice tied to a v1
  observation — r 16→32 for argument syntax, 3 epochs with a watched curve for
  abstention, checkpoints every 250 steps to pick the best rather than the last.

### Results (v2)
- v2 adapter (r=32, step 500/948, pushed by the 250-step Hub cadence after the
  VM was reclaimed) passes every acceptance threshold on the uncontaminated
  200-case set: emission 100%, name 100%, exact-args 76.5%, abstention 94%,
  parse 100% — mean 0.810 vs vanilla 0.255. v1's 0.855 on the same set is
  contaminated (those rows were its training data); on the mutually-clean
  12-voice set both adapters tie at 0.750. Full table: `docs/v2_report.md`.
- First measured training curve: val 0.281 → 0.844 over 500 steps.

### Results (clean verdict, 2026-08-23)
- Fresh 300-case set (unseen wording + unseen TTS engine, Kokoro held-out
  voices) synthesized LOCALLY in 4.7 min and pushed as the `test_fresh` split.
- Clean scores: vanilla 0.250 → v1 0.827 ≈ v2 0.833. Emission 99.6%, name
  98.7%, tolerant args 79.1%, abstention 96%, parse 100% — every acceptance
  threshold passed on fully held-out data. v1 and v2 tie once contamination is
  removed; v2 stays the deployable candidate on provenance.

### Conventions (from 2026-08-22)
- New and modified code comments are written in English.
- Work branches are named `rd/pr_rca_{action}` (action ≤ 2 words).
- Every push updates this changelog; pre-commit must pass on a clean test run.
