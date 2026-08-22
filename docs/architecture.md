# Architecture

## Découpage

Un seul paquet, `lfm2_audio`, découpé par **catégorie** puis par **concept
métier**. Les catégories transverses évitent de redéfinir les mêmes objets dans
chaque domaine.

| Sous-paquet | Rôle | Dépendances lourdes |
|---|---|---|
| `core/` | erreurs du domaine, environnement, format et rendu ChatML | aucune |
| `ds/` | structures de données | aucune (numpy seulement) |
| `serving/` | chargement du modèle, backends d'inférence | paresseuses |
| `vllm_plugin/` | plugin out-of-tree vLLM-Omni | vLLM, liquid-audio |
| `training/` | SFT LoRA, gel, export de checkpoint | torch, peft |
| `data_prep/` | génération et packing des datasets | datasets |
| `tools/` | outils métier appelables par le modèle | asyncpg (optionnel) |
| `orchestrator/` | boucle agent tool-calling, transport | fastapi (optionnel) |
| `rag/` | base de connaissances | chromadb |
| `evaluation/` | scoring des tool calls, métriques de latence | aucune |
| `cli/` | points d'entrée `lfm2-*` — argparse seulement | — |

**Règle de dépendance** : `core` et `ds` ne dépendent de rien du paquet ;
`serving` dépend de `core`/`ds` ; les CLIs dépendent de tout et rien ne dépend
d'elles. `serving.checkpoint` importe `training.export_checkpoint` et
`vllm_plugin.convert_checkpoint` **dans les fonctions** — ce sont des imports
lourds, et cela évite un cycle au niveau des paquets.

## pydantic ou dataclass ?

La règle est l'origine de la donnée, pas la préférence :

- **pydantic** pour ce qui franchit une frontière — configs YAML/env
  (`ds/config.py`), dialogues JSONL (`ds/dialogue.py`). La validation y a une
  valeur réelle et remplace du code de vérification manuel.
- **dataclass frozen** pour les objets construits par le code — `Waveform`,
  `Reply`, `ResolvedCheckpoint`. Valider un tableau numpy à chaque construction
  coûterait sans rien apprendre.

## Patterns du serving

Chacun est justifié par une pluralité **existante**, pas anticipée.

### Fabrique + registre — `serving/registry.py`

`BackendRegistry` associe un nom à un `BackendSpec` (module, classe,
dépendances). La classe n'est importée qu'à l'instanciation : `import lfm2_audio`
reste utilisable sur une machine sans GPU, et `backend="auto"` peut choisir le
premier backend réellement installé.

`LFM2Audio.from_pretrained` est la fabrique publique ; les sous-classes
implémentent `_build`.

### Chain of responsibility — `serving/checkpoint/sources.py`

Une spécification (`"exports/x"`, `"org/repo"`) est présentée à chaque
`CheckpointSource` dans l'ordre ; la première qui l'accepte la matérialise. Le
local passe avant le Hub, pour qu'un répertoire nommé `org/nom` gagne.

### Strategy — `serving/checkpoint/preparers.py`

Un cas de préparation = une classe : passthrough (déjà Omni), conversion
(layout liquid), fusion LoRA (base + adaptateur). `CheckpointResolver` choisit la
première applicable et ne connaît que l'interface.

### Template method — `serving/model.py`

`LFM2Audio.reply()` consomme `stream()` et concatène. Les backends
n'implémentent que `stream()` ; `reply`, `reset`, le contexte de gestion et les
métriques sont partagés.

## Invariants tenus par les objets

| Invariant | Porté par | Ce qu'il empêche |
|---|---|---|
| un signal ne se sépare pas de sa fréquence | `Waveform` | envoyer du 48 kHz à un mel calibré 16 kHz |
| au plus un tour porte de l'audio | `Conversation`, `ChatMLRenderer` | le bug multi-tours (N placeholders, 1 signal) |
| un checkpoint en cache est complet | marqueur `.lfm2_ready` | réutiliser un run interrompu |
| les chunks concaténés partagent leur fréquence | `Waveform.concat` | un rendu accéléré, sans erreur |

Ces invariants sont testés (`tests/test_waveform.py`,
`tests/test_conversation.py`, `tests/test_chatml_renderer.py`,
`tests/test_checkpoint_resolver.py`) — ce sont des régressions déjà rencontrées,
pas des hypothèses.

## Ce qui reste dette

`vllm_plugin`, `training`, `orchestrator`, `tools`, `data_prep`, `rag` et `cli`
sont vérifiés par mypy en mode relâché : ils manipulent massivement des objets
non typés en amont (couches vLLM, `nn.Module`, `Trainer` liquid-audio, handlers
FastAPI). La liste et sa justification sont dans `pyproject.toml`, à resserrer
module par module. `core`, `ds`, `serving` et `evaluation` sont en `strict` et ne
doivent pas régresser.
