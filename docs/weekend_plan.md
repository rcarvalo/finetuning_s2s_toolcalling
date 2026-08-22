# Weekend plan — EN baseline → dataset curation → fine-tune → re-eval

Compute: **Colab L4 22 GB first** (more quota), RunPod second.
Conventions: English comments, branches `rd/pr_rca_{action}`, changelog on every
push, pre-commit green on a clean test run.

| # | Step | Tooling | Status |
|---|---|---|---|
| 1 | EN audio baseline: WER, DNSMOS, NISQA | `notebooks/colab_baseline_eval.ipynb` → `lfm2-evaluate`, `benchmark/baseline_en/questions.jsonl` (24 q) | ready to run on L4 |
| 2 | EN tool-calling baseline | same notebook, `benchmark/toolcalling_en/cases.sample.jsonl` (12 cases), `tool_call` scorer | ready to run on L4 |
| 3 | Inventory `Rcarvalo/*` private datasets | `lfm2-dataset-inventory` (notebook cell or any logged-in machine) → `docs/dataset_inventory.md` | CLI ready — needs HF login |
| 4 | Curated dataset repo + enrichment pipeline | to design after step 3 results | pending |
| 5 | Preprocessing + push CLI to the curated repo | builds on `lfm2-preprocess-sft` / `lfm2-build-dataset` | pending |
| 6 | Training CLI from the baseline (steps/epochs sweeps) | builds on `lfm2-train-sft` + scoring callbacks | pending |
| 7 | Re-evaluate, compare to baselines, decide next steps | same two campaigns as 1–2, same question sets | pending |

## Step 1–2 how-to

Open `notebooks/colab_baseline_eval.ipynb` on an L4 runtime and run top to
bottom. Reports land in `reports/baseline_en_audio.json` and
`reports/baseline_en_toolcalling.json`; download them and commit them to the
branch. NISQA weights are optional (`NISQA_MODEL_PATH`); without them the
campaign still measures WER + DNSMOS and marks NISQA `unavailable`.

**Comparability rule**: later fine-tunes are compared on the *same* question
sets, same `max_tokens`, same ASR model (`openai/whisper-large-v3-turbo`).
The report's `context` block records all of this — never compare two reports
whose contexts differ.

## Known constraints

- L4 22 GB fits the 1.5B model + Whisper large-v3-turbo comfortably (bf16).
- DNSMOS ONNX weights are fetched from the microsoft/DNS-Challenge repo at
  runtime (not redistributable in git).
- `.env` is not in git; recreate from `.env.example` (HF_TOKEN needed locally
  for steps 3–5).
