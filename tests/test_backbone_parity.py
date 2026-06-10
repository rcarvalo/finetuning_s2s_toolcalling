"""Test de parité P0 : backbone exporté (layout HF Lfm2ForCausalLM) vs liquid-audio.

Valide que l'implémentation Lfm2 standard (transformers — même code modèle que
vLLM core côté définitions HF) reproduit les logits du backbone liquid-audio sur
des prompts outillés, AVANT d'investir dans les stages vLLM-Omni (cf.
docs/vllm_omni_integration.md, phase P0).

GPU requis + checkpoint exporté :

    python -m s2s_toolcalling.training.export_checkpoint \\
        --base LiquidAI/LFM2.5-Audio-1.5B --mode backbone --output exports/backbone
    EXPORTED_BACKBONE=exports/backbone BASE_MODEL=LiquidAI/LFM2.5-Audio-1.5B \\
        python -m pytest tests/test_backbone_parity.py -m gpu -q
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("liquid_audio")
pytest.importorskip("transformers")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif("EXPORTED_BACKBONE" not in os.environ, reason="EXPORTED_BACKBONE not set"),
]

# Prompts outillés représentatifs (system + tool list + tour user → tool call attendu).
PROMPTS = [
    '<|startoftext|><|im_start|>system\nTu es l\'assistant d\'accueil. Liste des outils disponibles : '
    '<|tool_list_start|>[{"name":"check_appointment"}]<|tool_list_end|><|im_end|>\n'
    "<|im_start|>user\nBonjour, je suis Marie Dupont, j'ai rendez-vous avec Claire Martin.<|im_end|>\n"
    "<|im_start|>assistant\n",
    "<|startoftext|><|im_start|>user\nQuel est le wifi invité ?<|im_end|>\n<|im_start|>assistant\n",
]


@pytest.fixture(scope="module")
def models():
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = os.environ.get("BASE_MODEL", "LiquidAI/LFM2.5-Audio-1.5B")
    exported = os.environ["EXPORTED_BACKBONE"]

    liquid = LFM2AudioModel.from_pretrained(base, device="cuda", dtype=torch.bfloat16).eval()
    proc = LFM2AudioProcessor.from_pretrained(base, device="cuda").eval()
    hf = AutoModelForCausalLM.from_pretrained(exported, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(exported)
    return liquid, proc, hf, tok


@torch.no_grad()
def test_logits_parity(models):
    liquid, proc, hf, tok = models
    for prompt in PROMPTS:
        ids = proc.text.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()
        assert tok.encode(prompt, add_special_tokens=False) == ids[0].tolist(), "tokenizers diverge"

        # liquid-audio : embeddings texte → backbone → logits via embeddings liés
        emb = liquid.lfm.embed_tokens(ids)
        liquid_hidden = liquid.lfm(inputs_embeds=emb).last_hidden_state
        liquid_logits = torch.nn.functional.linear(liquid_hidden[0, -1], liquid.lfm.embed_tokens.weight)

        hf_logits = hf(input_ids=ids).logits[0, -1]

        top_liquid = liquid_logits.topk(10).indices
        top_hf = hf_logits.topk(10).indices
        assert torch.equal(top_liquid, top_hf), f"top-10 tokens diverge on prompt: {prompt[:60]}..."
        assert torch.allclose(liquid_logits.float(), hf_logits.float(), atol=0.5, rtol=1e-2)


@torch.no_grad()
def test_greedy_continuation_parity(models):
    liquid, proc, hf, tok = models
    prompt = PROMPTS[0]
    ids = proc.text.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()

    hf_out = hf.generate(ids, max_new_tokens=32, do_sample=False)
    hf_text = tok.decode(hf_out[0, ids.shape[1] :])

    cur = ids
    for _ in range(32):
        emb = liquid.lfm.embed_tokens(cur)
        hidden = liquid.lfm(inputs_embeds=emb).last_hidden_state
        logits = torch.nn.functional.linear(hidden[0, -1], liquid.lfm.embed_tokens.weight)
        nxt = logits.argmax().view(1, 1)
        cur = torch.cat([cur, nxt], dim=1)
    liquid_text = proc.text.decode(cur[0, ids.shape[1] :])

    assert liquid_text == hf_text
