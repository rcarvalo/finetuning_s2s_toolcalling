# Datasets — what exists, what it contains, how it flows

Reference for the English tool-calling work. Everything here was read from the
Hub or from the files themselves, not from memory; the raw catalogue of all 50
repos is `docs/dataset_inventory.md` (regenerate with `lfm2-dataset-inventory`).

---

## 1. The common format

Every stage of the pipeline speaks the same JSONL **dialogue schema**
(`lfm2_audio.ds.dialogue`, pydantic-validated), so one file can be synthesized,
packed for training and replayed as an evaluation set without conversion.

```jsonc
{
  "id": "tc_000000_web_search",
  "tools": ["web_search", "db_query"],
  "meta": {"target": "web_search", "style": "with disfluency", "depth": "requires inference"},
  "turns": [
    {"role": "user",      "text": "…what's the latest news on the elections?",
                          "audio": "tc_000000_web_search_u0.wav", "voice": "casual_female"},
    {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "…"}}]},
    {"role": "tool",      "content": {"results": "…"}},
    {"role": "assistant", "text": "Here's what I found…"}
  ]
}
```

Invariants the schema enforces (they are not cosmetic — each caught a real bug):

- `audio` is a path **relative** to the dataset's audio root, never a signal.
- A tool-call turn carries **no audio**: tool calls are emitted in the text
  stream. "Think in text, speak in audio."
- A `user` turn must carry `text` or `audio`; a `tool` turn must carry `content`.
- `voice` records which TTS voice produced the turn — that is what lets us ask
  whether a model generalizes across timbres rather than memorizing one.
- Unknown fields are refused, so a typo fails loudly instead of being ignored.

**Single-turn (Phase A)** = 2 turns, user → tool call. Teaches *the decision*.
**Conversational (Phase B)** = 4 turns, the tool result is reinjected and the
assistant answers out loud. Teaches *the spoken answer after the round-trip*.

---

## 2. The three on-topic datasets

Of 50 repos under `Rcarvalo/`, three serve English tool calling.

### `Rcarvalo/tc-en-audio-toolcalling` — private, 477 MB — **the training set**

Synthetic spoken English, two tools (`web_search`, `db_query`), single-turn.
Audio is 16 kHz mono, synthesized with Voxtral TTS.

| Split | Rows | Tool distribution | Voices |
|---|---:|---|---|
| train | 2930 | db_query 1106 · web_search 1072 · none 752 | casual_male 991 · cheerful_female 975 · casual_female 964 |
| test | 12 | 4 · 4 · 4 | **neutral_female 8 · neutral_male 4** (held out from train) |

Flat columns on the Hub (`audio`, `utterance`, `has_tool_call`, `tool_name`,
`arguments`, `assistant_text`, `expected_calls`, `voice`, `target`, `style`,
`depth`) and rehydrated into the dialogue schema by `lfm2-hf-to-dialogues`.

The `none` rows are **negatives**: chit-chat where the right answer is to reply
directly. They are what keeps a fine-tune from learning "always call something".

### `Rcarvalo/tc-en-s2s-src` — private, 3.1 MB — **conversational, text only**

3000 dialogues in the same schema, with the full round-trip:
2197 four-turn (a tool is called, its result is reinjected, the assistant
answers) and 803 two-turn negatives. Targets: db_query 1157 · web_search 1040 ·
none 803. Also present locally as `data/tc_en_s2s.jsonl`.

**No audio has ever been produced for it.** Each dialogue would yield a Phase A
example (user → tool call) *and* a Phase B one (result → spoken answer), so
synthesizing it roughly doubles the corpus and unlocks Phase B in one run.

### `Rcarvalo/tc-en-s2s-audio` — public — **empty**

Only `.gitattributes`. The TTS run for the conversational set was never pushed.

---

## 3. The curated corpus we train on

### `Rcarvalo/tc-en-voice-agent-v1` — private — built by `lfm2-dataset-repack`

Derived from `tc-en-audio-toolcalling` because its 12-row test split cannot
separate a real improvement from noise.

| Split | Rows | Composition | What it answers |
|---|---:|---|---|
| `train` | 2729 | db 1030 · ws 999 · none 700 | — |
| `test_utterances` | 200 | db 76 · ws 73 · none 51 | unseen wording, **seen voices** → statistical power |
| `test_voices` | 12 | 4 · 4 · 4, neutral_* | **unseen voices** → generalization across timbre |

Built by: dropping duplicate utterances first (so copies of one utterance cannot
straddle the boundary), then a **stratified** split on the tool actually called
— read from the assistant turn, not from `meta`, which can be stale. The tool
distribution is preserved within one point of the source (38/36/26 vs
37.7/36.6/25.7), and the split is deterministic for a given seed.

**The two evaluation splits must never be merged**: they measure different
things, and averaging them hides which one moved.

---

## 4. Evaluation sets

| Set | Cases | Input | Used for |
|---|---:|---|---|
| `benchmark/baseline_en/questions.jsonl` | 24 | text prompt, spoken answer | audio quality: WER, DNSMOS, NISQA |
| `benchmark/toolcalling_en/cases.sample.jsonl` | 12 | **text only** | the published 0.333 baseline |
| `test_utterances` / `test_voices` | 200 / 12 | **real audio** | speech → tool call |

> The 12 benchmark cases carry no `audio` field, so the published tool-calling
> baseline measured **text → tool call**. The conclusion holds a fortiori (text
> is easier, and the model emitted nothing), but it is not the audio figure.

---

## 5. Trained adapters already on the Hub

| Repo | Size | Config | Trained on |
|---|---:|---|---|
| `Rcarvalo/lfm25-tc-en-adapter` (private) | 22 MB | LoRA r=16, targets `q/k/v/out/in_proj` **and** `w1/w2/w3` | the original 2930-row train split (2026-06-16) |
| `Rcarvalo/lfm25-tc-en-s2s-adapter` (public) | 22 MB | same | Phase B (2026-06-21) |
| `Rcarvalo/lfm25-tc-en-s2s-omni` (private) | 3.6 GB | merged + converted for vLLM-Omni serving | — |

⚠️ **Contamination note.** `test_utterances` was carved out of the same 2930
rows that trained the June adapter, so for *that* adapter those 200 cases are
training data. Its only honest held-out set is `test_voices` (12 cases). Future
runs trained on `tc-en-voice-agent-v1/train` can use the 200 cleanly.

---

## 6. Everything else in the account (~700 GB, French)

Not used here, listed so nobody re-derives it: bilingual pilots
(`lfm2-bilingual-pilot-125h` 23.6 GB, `lfm2-bilingual-fr-partial` 76 GB), French
TTS corpora (`french-dialogue-tts-1000h` 61 GB, `-100h`,
`emilia-yodas-fr-filtered`, `kanitts2-fr-*`, `audioFRv1`,
`lfm2-audio-fr-data`), large unlabelled pools (`audio_dataset` 407 GB,
`audio-import` 188 GB), the ~20 `hugodecrypte_*` chunks, and TTS benchmarks
(`benchmarkTTS`, `tts-benchmark-results`, `vibevoice-lora-siwis-samples`).

These matter for the French stage of the project. Their contents were **not**
inspected — only names, sizes and tags — so treat the grouping above as a map,
not an inventory of what is inside.

---

## 7. How data flows

```
generate (LLM)            data/tc_en_train.jsonl        text dialogues
      │                   data/tc_en_s2s.jsonl
      ▼
lfm2-synthesize-audio     + WAV per turn, voice recorded     ← Voxtral TTS
      ▼
lfm2-build-dataset        → Hub (parquet, Audio feature)
      ▼
lfm2-dataset-repack       → tc-en-voice-agent-v1 (train / test_utterances / test_voices)
      ▼
lfm2-hf-to-dialogues      → JSONL + WAV on the training box
      ▼
lfm2-dataset-curate       merge sources, dedup, refuse held-out leakage
      ▼
lfm2-preprocess-sft       → packed tensors  →  accelerate launch … train.sft
      ▼
lfm2-evaluate             → reports/*.json  →  lfm2-eval-compare
```

`lfm2-dataset-curate` is the guard rail: it deduplicates on the normalized
utterance (case, accents and punctuation folded; apostrophes deleted so a
contraction matches its spelled-out form) and **exits 1** if any dialogue also
appears in the held-out split. Its first real run merged 5930 → 5788 dialogues:
142 duplicates removed, **0 leakage** against the benchmark — which is what
retroactively confirms the baseline was measured on uncontaminated data.

## 8. What is missing

1. **Conversational audio** — `tc-en-s2s-src` is still text. Blocked on the
   Voxtral environment (see `docs/pre_training_review.md` §5), not on tooling.
2. **Voice diversity** — three voices in training. `test_voices` exists to
   detect over-fitting to timbre, and it is only 12 cases wide.
3. **Two tools only** — a binary choice can be won without learning to abstain;
   relevance on negatives is the metric that catches it.
4. **No audio augmentation** — training audio is clean TTS, the Reachy Mini will
   hear a room.
