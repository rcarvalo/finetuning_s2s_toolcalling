# Entraîner

`lfm2_audio.training` **enveloppe** le `Trainer` officiel de liquid-audio, il ne
le réimplémente pas : la boucle, Accelerate, le scheduler et les checkpoints
restent en amont. On y ajoute deux choses seulement.

```bash
accelerate launch -m lfm2_audio.cli.train.sft --config configs/training/phase_en_toolcalling.yaml
lfm2-train-sft --config <recette>.yaml --print-config   # valide sans rien lancer
```

## Ce que le wrapper ajoute

1. **le clipping de gradient**, absent de la boucle amont — sans lui,
   l'entraînement peut diverger sur un batch aberrant ;
2. **l'émission d'événements** vers des callbacks.

Tout le reste — journal, wandb, sauvegardes, push Hub, scoring périodique — vit
dans des callbacks activés par la configuration. La boucle d'optimisation reste
celle d'amont, ce qui permet de suivre les versions de liquid-audio sans
réécrire l'entraînement.

## Les callbacks

| Callback | Activé par | Rôle |
|---|---|---|
| `ConsoleCallback` | toujours | loss, perplexité texte, grad norm, lr |
| `ScoringCallback` | `evaluation.enabled` | les scorers d'éval tous les N pas |
| `WandbCallback` | `wandb_project` | publie toutes les métriques du contexte |
| `CheckpointCallback` | toujours | `accelerator.save_state` tous les N pas |
| `HubPushCallback` | `hub_repo` | pousse l'adaptateur LoRA |

**L'ordre compte** et fait partie du contrat : les callbacks qui *produisent* des
métriques passent avant ceux qui les *publient*. `StepContext.metrics` est le
tableau de bord partagé de l'événement en cours — un scorer y dépose, wandb y
lit. C'est `CallbackBuilder` qui garantit cet ordre.

Un callback qui échoue est signalé puis neutralisé : perdre le suivi wandb ne
doit pas perdre l'entraînement.

## Ajouter un callback

```python
from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext


class EarlyStopOnWer(TrainingCallback):
    """Arrête si le WER remonte — la dérive des têtes audio se voit là en premier."""

    def __init__(self, ceiling: float = 0.20) -> None:
        self._ceiling = ceiling

    def on_step_end(self, context: StepContext) -> None:
        wer = context.metrics.get("score/wer")
        if wer is not None and wer > self._ceiling:
            raise RuntimeError(f"WER {wer:.3f} au-dessus de {self._ceiling}")
```

## LoRA et gel

L'injection LoRA et les politiques de gel s'appliquent **avant** que le Trainer
ne crée l'optimiseur, via un hook sur `from_pretrained` — c'est l'unique fenêtre
offerte par le Trainer amont, et le seul moment où geler un module a l'effet
attendu sur les groupes de paramètres.

Avec `lora.enabled: true`, le backbone est gelé : seuls les adaptateurs
s'entraînent, et AdamW ignore les paramètres sans gradient.

## Le ratio interleaved

Calibré par `lfm2-calibrate`, il est écrit dans le `config.json` du checkpoint
exporté. C'est la **source unique** entre entraînement et serving : ne pas le
dupliquer dans le code.
