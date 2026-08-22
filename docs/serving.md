# Servir LFM2.5-Audio

## Charger un modèle

```python
from lfm2_audio import LFM2Audio

model = LFM2Audio.from_pretrained("Rcarvalo/lfm25-tc-en-s2s")
text, audio = model.reply(audio="question.wav")
audio.save("reponse.wav")
```

`from_pretrained` fait tout le travail de résolution. Ce qui manque est produit
une fois puis mis en cache (`$LFM2_SERVE_CACHE`, défaut
`~/.cache/lfm2_audio/checkpoints`), avec un marqueur de fin : un run interrompu
n'est jamais réutilisé tel quel.

| Ce que vous passez | Ce qui se passe |
|---|---|
| répertoire au layout Omni | utilisé tel quel |
| répertoire au layout liquid-audio | `config.json` réécrit au layout Omni |
| repo Hugging Face | téléchargé, puis re-détecté |
| base + `adapter=` | LoRA fusionné (GPU) puis converti |
| répertoire d'adaptateur seul | la base est lue dans `adapter_config.json` |

Un backbone texte seul (`Lfm2ForCausalLM`) est **refusé** avec un message
explicite : il est servable par `vllm serve`, mais sans audio.

## Choisir un backend

```python
LFM2Audio.from_pretrained(ckpt, backend="vllm")  # 2 stages, streaming, TTFA ~300 ms
LFM2Audio.from_pretrained(ckpt, backend="liquid")  # référence PyTorch, batch = 1
LFM2Audio.from_pretrained(ckpt, backend="auto")  # premier installé (défaut)
```

`liquid` sert d'étalon de parité numérique et dépanne là où vLLM-Omni ne
s'installe pas. Il re-préfille tout le contexte à chaque tour : nettement plus
lent en dialogue.

## Streaming

```python
for chunk in model.stream(audio="question.wav"):
    play(chunk.samples)  # Waveform 24 kHz
print(model.last_reply.metrics.ttfa_s)
```

`reply()` n'est que `stream()` consommé puis concaténé — mêmes métriques, même
historique.

## Dialogue multi-tours

L'historique est tenu par le modèle ; `reset()` repart d'un contexte neuf.

```python
model.reply(audio="tour1.wav")
model.reply(text="et demain ?")  # le contexte du tour 1 est conservé
model.reset()
```

L'audio d'un tour n'est fourni au modèle **qu'une fois** : ensuite le tour est
conservé en texte (`(voice message)`). C'est délibéré — `multi_modal_data` ne
transporte que le signal courant, et un second placeholder ferait scatter
l'audio sur une position périmée.

## Régler la latence

Le YAML par stage (`configs/serving/vllm_omni.yaml`) est le chemin recommandé :
CUDA graphs PIECEWISE sur le stage 0 et `initial_codec_chunk_frames=2`, soit un
TTFA de 250-350 ms contre ~750 ms en tout-eager.

```python
from lfm2_audio.ds.config import EngineConfig

LFM2Audio.from_pretrained(ckpt, engine=EngineConfig())  # YAML (défaut)
LFM2Audio.from_pretrained(ckpt, engine=EngineConfig(deploy_config=None))  # legacy eager
```

Mesurer avant de conclure :

```bash
lfm2-bench --checkpoint exports/lfm25_audio_fr_omni --runs 5
```

Deux défauts ne doivent pas être touchés sans lire
[`optimization_audit.md`](optimization_audit.md) :

- `enable_prefix_caching=False` — le chemin `omni_prefix_cache` du runner perd
  l'export sparse `codes.audio` : texte correct, **zéro** chunk vers le stage 1 ;
- `async_scheduling=False` — l'async tronque d'un token l'historique que le
  sampler rejoue pour reconstruire la modalité.

## Diagnostiquer une absence d'audio

Deux pannes se ressemblent en sortie ; `lfm2-bench` et les logs du backend les
distinguent :

| Symptôme | Cause | Où chercher |
|---|---|---|
| 0 frame émise par le stage 0 | pas de `<|text_end|>` | prompt / system / checkpoint |
| frames émises, 0 chunk reçu | plomberie connector stage 0 → 1 | `LFM2_DEBUG_CHUNK=1` |

## Tool calling

L'orchestrateur a besoin des points d'arrêt et de `stream_turns`, que seul le
backend vLLM expose :

```python
from lfm2_audio.orchestrator.vllm_tool_agent import VllmToolAgent
from lfm2_audio.serving.backends.vllm_omni import VllmOmniBackend

backend = VllmOmniBackend.from_pretrained(ckpt, backend="vllm")
agent = VllmToolAgent(backend, registry)
for event in agent.respond(Waveform.from_file("question.wav")):
    ...
```

Démo clé en main : `lfm2-toolcalling-demo --checkpoint … --share`.
