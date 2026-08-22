# Évaluer un modèle

Une campagne mesure deux choses distinctes, avec le même jeu de questions :
la **qualité de l'audio** produit (est-il intelligible, agréable ?) et la
**qualité de la réponse** (appelle-t-il le bon outil, dit-il quelque chose qui
découle du résultat ?).

```bash
lfm2-evaluate --checkpoint exports/tc_en \
              --questions benchmark/toolcalling_en/cases.sample.jsonl \
              --out reports/tc_en.json
lfm2-evaluate --list-scorers          # ce qui est mesurable sur cette machine
```

## Les métriques

| Scorer | Mesure | Sens | Dépendances |
|---|---|---|---|
| `wer` | audio généré re-transcrit vs. ce qu'il devait dire | ↓ | `torch`, `transformers` |
| `dnsmos` | MOS P.835 prédit (`sig`/`bak`/`ovrl`), sans référence | ↑ | `onnxruntime` + poids |
| `nisqa` | MOS NISQA v2 prédit, sans référence | ↑ | `torch` + poids |
| `tool_call` | tour réussi : bon appel, ou abstention justifiée | ↑ | — |
| `reasoning` | réponse notée par un juge LLM sur rubrique versionnée | ↑ | `google-genai` + clé |

`wer` est le signal le plus **précoce** d'une dérive des têtes audio : il bouge
avant qu'un MOS ne bouge. `dnsmos` et `nisqa` sont complémentaires — la première
vise le débruitage, la seconde les dégradations de transmission ; sur de la
parole synthétique elles se trompent rarement de la même façon.

`tool_call` agrège le **tour réussi**, pas `call_correct` brut : cette facette
est fausse par construction sur un cas négatif, si bien que l'agréger seule
punirait chaque abstention correcte.

## Métriques indisponibles

Les poids DNSMOS et NISQA ne sont pas redistribuables. Sans eux, le scorer se
déclare `unavailable` avec la marche à suivre — **la campagne continue** :

```
  wer         0.084 ↓
  dnsmos          —   modèle DNSMOS introuvable — poser son chemin dans $DNSMOS_MODEL_PATH
  tool_call   0.917 ↑
```

Le rapport distingue donc « non mesuré faute d'outillage » de « mesuré et
mauvais » — deux conclusions très différentes.

```bash
uv sync --extra eval
export DNSMOS_MODEL_PATH=/chemin/sig_bak_ovr.onnx   # microsoft/DNS-Challenge
export NISQA_MODEL_PATH=/chemin/nisqa.tar           # gabrielmittag/NISQA
export GEMINI_API_KEY=…                             # juge du scorer `reasoning`
```

`--fail-on-unavailable` inverse ce comportement quand la campagne doit être
complète ou rien (CI de release).

## Suivre les mêmes métriques pendant l'entraînement

C'est le point de toute l'architecture : **les mêmes objets**. Le WER affiché au
pas 500 et celui du rapport final sortent du même code, donc ils sont
comparables.

```yaml
# configs/training/<recette>.yaml
evaluation:
  enabled: true
  question_set: benchmark/toolcalling_en/cases.sample.jsonl
  interval: 500
  at_start: true          # mesure de référence : sans elle rien n'est lisible
  max_questions: 32       # garder petit : générer puis transcrire coûte cher
  scoring:
    scorers:
      - name: tool_call
        options: { arg_match: token_f1, threshold: 0.7 }
      - name: wer
```

Les valeurs remontent dans wandb sous `score/*`, sans une ligne de Python.

## Écrire un scorer

Une classe, deux méthodes utiles :

```python
from typing import ClassVar
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample


class SpeakingRateScorer(BaseScorer):
    name = "speaking_rate"
    higher_is_better: ClassVar[bool] = True

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio and bool(sample.predicted_text)

    def measure(self, sample: EvalSample) -> ScoreResult:
        words = len(sample.predicted_text.split())
        return ScoreResult.ok(self.name, words / sample.predicted_audio.duration_s)
```

Puis l'enregistrer :

```python
from lfm2_audio.scorer.registry import SCORERS
from lfm2_audio.scorer.spec import ScorerSpec

SCORERS.register(
    ScorerSpec(
        name="speaking_rate",
        module="mon_projet.speaking_rate",
        class_name="SpeakingRateScorer",
    )
)
```

`score()` n'est pas à surcharger : c'est une *template method* qui traite la
disponibilité, le hors-périmètre et les exceptions. Un scorer ne fait donc
**jamais** échouer une campagne de 500 cas.
