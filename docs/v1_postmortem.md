# v1 post-mortem — why the June adapter looked dead, and what it had learned

Three mechanisms were inert or misleading. None of them was the model.

## The three failures

**1. The evaluation never declared the tools.** `lfm2-evaluate` built the model
with `DEFAULT_SYSTEM` ("Respond with interleaved text and audio"), which names
no tool, while training embeds the definitions in the system prompt. Every
campaign asked the model to call tools it had never been told existed — base
model and fine-tune alike scored zero emissions. *Fix:* `--tool-definitions`
renders the training prompt byte for byte (asserted by a test), and the choice
is recorded in the report context.

**2. Interleaved audio decode shredded the spans.** With tools declared, the
adapter emitted calls — but they read as mangled syntax:
`web_search( php="vegetarian recipe")`, `db_query(question terms=…)`. That is
the interleaving inserting audio placeholders inside a structured span, exactly
what the `GenerationConfig.text_only` docstring warned about. *Fix:* campaigns
that grade no audio decode text-only, automatically.

**3. Training ran blind.** The train CLI never passed a `generator_factory`, so
`ScoringCallback` silently resolved to no generator and skipped every
measurement. Nobody could see during training that anything was off. *Fix:*
`training/eval_generator.py` wraps the live model in the same serving stack as
the final campaign (text-only, tools declared, eval-mode toggled and restored).

## What v1 actually learned, measured after the fixes

| | held-out voices (12) | seen voices (60) |
|---|---|---|
| calls emitted | 8/12 | **60/60** |
| routing (which tool) | correct | correct |
| abstention on negatives | — | **lost** (calls on everything) |

So: routing learned, generalizes to unseen voices reasonably; the real
regression is **abstention**. The v2 recipe answers each line
(`configs/training/tc_en_voice_agent_v2.yaml`).

## The meta-lesson

All three failures share a shape: **a mechanism that silently degrades to
"nothing" instead of failing loudly** — a prompt without tools still generates,
a shredded span still parses as text, a `None` factory still trains. Where a
default can quietly produce a meaningless measurement, prefer an explicit
argument and a log line; where a callback can skip, log the skip. The scoring
callback now logs its generator; the eval CLI logs its decode mode and records
the tool declaration in the report context.
