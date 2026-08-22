# lfm2-audio

Fine-tuning, serving vLLM et orchestration tool-calling de LFM2.5-Audio-1.5B
(assistant vocal S2S français + anglais, embarqué dans Reachy Mini).

## Stack

Python 3.12 · uv · pydantic · pytest · ruff + mypy · vLLM-Omni 0.22 (GPU) ·
liquid-audio (référence)

## Écarts assumés avec les règles globales

- **Layout `python/` et non `src/`** — choix explicite : sous-paquets par
  catégorie (`core/`, `ds/`, `cli/`, puis les concepts métier).
- **mypy `strict` scopé** — il tient sur `core`, `ds`, `serving`, `evaluation` et
  ne doit pas y régresser. Les paquets antérieurs (`vllm_plugin`, `training`,
  `orchestrator`, `tools`, `data_prep`, `rag`, `cli`) sont vérifiés en mode
  relâché ; la liste est dans `pyproject.toml` avec sa justification.

## Commandes

```bash
make install / install-serving / hooks
make check        # lint-check + typecheck + test — ce que vérifie la CI
make test         # pytest -m "not gpu"
uv run pytest tests/test_waveform.py -q          # un seul fichier
uv run lfm2-demo --checkpoint <ckpt> --text "hi" # une CLI sans réinstaller
```

Aucun GPU sur le Mac de dev : tout ce qui touche vLLM-Omni ou liquid-audio se
vérifie sur Colab / pod. Dire explicitement ce qui n'a pas pu être exécuté.

## Architecture — ce qui n'est pas évident à la lecture

- **`import lfm2_audio` doit rester léger.** Ni torch, ni vLLM, ni liquid-audio à
  l'import du paquet : le plugin est chargé dans *chaque* process worker vLLM.
  D'où le `__getattr__` PEP 562 dans `__init__.py`, le registre de backends qui
  importe par chaîne, et les imports lourds placés dans les fonctions.
- **`serving/backends/omni_engine.py` concentre les contournements vLLM 0.22.**
  Chacun a été mesuré ; les retirer casse le flux audio *silencieusement* (le
  texte continue de sortir, plus aucun chunk n'arrive). Idem pour les défauts de
  `EngineConfig` (`enable_prefix_caching=False`, `async_scheduling=False`).
- **Un seul tour peut porter de l'audio.** `Conversation` tient l'invariant ;
  `ChatMLRenderer` refuse d'en rendre deux. C'était le bug multi-tours : N
  placeholders pour un seul signal → l'audio scatte sur une position périmée et
  le modèle n'entend plus rien au-delà du tour 1.
- **`Waveform` transporte toujours sa fréquence.** L'encodeur mel est calibré
  16 kHz, le détokeniseur Mimi sort du 24 kHz ; un mélange ne lève aucune erreur,
  il dégrade juste les réponses. Ne jamais revenir à `tuple[ndarray, int]` nu.
- **`reply.text` vs `reply.raw_text`** : `text` est parlable (marqueurs retirés),
  `raw_text` garde les `<|tool_call_*|>` dont l'orchestrateur a besoin.
- **Les CLIs sont rangées par domaine** : `cli/data/` (dataset), `cli/train/`,
  `cli/eval/`, `cli/serve/`. Un module de `cli/` ne porte QUE son argparse — la
  logique vit dans le sous-paquet métier, testable sans CLI. Les noms des
  commandes (`lfm2-*` dans `[project.scripts]`) ne bougent pas quand un module
  se déplace.
- **`remote/protocol.py` est le contrat de fil**, importé par le client ET par
  `infra/handler.py`. Tout ce qui vient du réseau y est validé par pydantic
  (union discriminée sur `kind`) : aucun accès `dict` brut sur une réponse. Un
  champ modifié casse les deux bords en même temps — c'est voulu.

## Gotchas

- vLLM 0.22 est un wheel **CUDA 13** : sur un hôte CUDA 12 (Colab), utiliser le
  wheel `+cu129` — `lfm2_audio.core.env.require_vllm()` affiche la commande.
- `peft` reste `<0.15` : au-delà, `torchao>=0.16` est ininstallable avec le torch
  de l'env vLLM 0.22.
- Le ratio interleaved calibré vit dans le `config.json` du checkpoint exporté —
  c'est la source unique entraînement/serving, pas une constante du code.
- Données brutes et checkpoints ne vont jamais dans git (cf. `.gitignore`).
