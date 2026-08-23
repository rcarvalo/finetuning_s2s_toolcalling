# Infra RunPod — inférence serverless + entraînement batch

Deux chemins indépendants :

| Besoin | Entrée | Coût |
|---|---|---|
| Inférence S2S depuis le poste local / le Reachy Mini | endpoint **serverless** (scale-to-zero) | à la seconde d'exécution |
| Batch d'entraînement lancé depuis le poste local | **pod** éphémère via SkyPilot | durée du job, pod détruit ensuite |

## 1. Inférence serverless (façon NIM)

`RUNPOD_ENDPOINT_ID` n'existe **qu'après** la création de l'endpoint : il faut
d'abord une image, puis un endpoint qui la sert. Deux images, deux usages :

| Dockerfile | Backend | Pour quoi | Taille |
|---|---|---|---|
| `Dockerfile.serve.liquid` | liquid-audio (référence) | **premier déploiement** : valider la plomberie, flux audio déjà prouvé | ~6 Go |
| `Dockerfile.serve` | vLLM-Omni | production : TTFA ~300 ms, prefix caching | ~12 Go |

### Chemin recommandé — build par RunPod depuis GitHub (aucun Docker local)

Le Mac de dev est en arm64 et l'image doit être linux/amd64 : cross-builder puis
pousser ~6 Go depuis la maison est lent. RunPod build à notre place.

1. Pousser la branche sur GitHub (le Dockerfile doit y être).
2. Console RunPod → **Serverless → New Endpoint → GitHub Repo**, autoriser
   l'app GitHub, choisir le repo + la branche.
3. **Dockerfile Path** : `infra/Dockerfile.serve.liquid`.
4. GPU **24 Go** (RTX 4090 / A5000), **flex** min 0 / max 1, FlashBoot activé,
   **idle timeout 5–10 s** pendant les tests (30–60 s en usage conversationnel),
   datacenter **EU**.
5. Copier l'**Endpoint ID** affiché → `RUNPOD_ENDPOINT_ID=` dans `.env`.

Contraintes du flux GitHub : ni `--build-arg` ni secret de build, build ≤ 30 min
par étape Docker, image ≤ 80 Go. D'où une image figée sur le modèle **public**
`LiquidAI/LFM2.5-Audio-1.5B` — on choisit la variante par le chemin du
Dockerfile, pas par des arguments.

### Chemin alternatif — build local (checkpoint privé, arguments)

Nécessaire dès qu'on sert un checkpoint privé (le token HF passe alors en
secret de build, jamais dans une layer) :

```bash
export HF_TOKEN=...
docker build --platform linux/amd64 -f infra/Dockerfile.serve \
    --build-arg LFM2_CHECKPOINT=Rcarvalo/<repo-finetuné> \
    --build-arg LFM2_ADAPTER=Rcarvalo/lfm25-tc-en-adapter \
    --secret id=hf_token,env=HF_TOKEN \
    -t <docker-user>/lfm2-audio-serverless:v1 .
docker push <docker-user>/lfm2-audio-serverless:v1
```

Puis même création d'endpoint, en choisissant **Docker Image** au lieu de
GitHub Repo.

Dans les deux cas les poids sont téléchargés (et convertis au layout Omni pour
l'image vLLM) **pendant le build** par `bake_checkpoint.py` : le cold start ne
paie ni download ni conversion.

### Appeler depuis le poste local ou le Reachy

```bash
uv sync --extra client         # httpx seulement, aucune dépendance GPU
```

```python
from lfm2_audio.remote import LiquidAudioClient

llm = LiquidAudioClient("<endpoint-id>")  # RUNPOD_API_KEY lu depuis l'env
text, audio = llm.invoke(audio="question.wav")  # bloquant
audio.save("reponse.wav")

for chunk in llm.invoke_stream(audio="question.wav"):
    play(chunk)  # le robot parle dès le 1er chunk
```

Contrat d'événements (handler → client) : `{"kind": "audio", "audio_b64",
"sample_rate"}` par chunk, puis `{"kind": "final", "text", "raw_text",
"metrics"}`. v1 **stateless** : un job = un tour, contexte remis à zéro.

### Tester à la voix sans dépasser ~1 $ : `make app`

`app/gradio_app.py` — micro → endpoint → réponse audio streamée + texte +
métriques (TTFA, coût estimé par tour et cumulé). L'app **refuse d'envoyer**
au-delà de `APP_BUDGET_USD` (défaut 1 $ ; estimation au temps mur, borne
haute). Pour que le compteur colle à la facture : endpoint en **flex, max
workers 1, idle timeout court (5–10 s)** pendant les tests. Ordre de grandeur
RTX 4090 flex : ~0,005 $ le tour → un budget de 1 $ ≈ 100+ tours.

## 2. Entraînement batch (SkyPilot)

```bash
uv tool install "skypilot[runpod]"
runpod config    # colle la clé API
sky check runpod

set -a; source .env; set +a
sky launch -c liquid-train infra/sky_train.yaml --down \
    --env HF_TOKEN --env WANDB_API_KEY
```

`sky launch` provisionne le pod, sync le repo, `uv sync --frozen --extra train`
(versions verrouillées par `uv.lock`), prépare les données
(`prepare_data.py`, idempotent) puis lance `lfm2-train-sft`. `--down` détruit
le pod à la fin — l'adaptateur LoRA est poussé sur le Hub tous les
`push_interval` steps (cf. config d'entraînement), donc rien n'est perdu.

Commandes utiles :

```bash
sky exec liquid-train infra/sky_train.yaml --env HF_TOKEN   # relancer un batch
ssh liquid-train                                            # shell sur le pod
sky queue liquid-train ; sky logs liquid-train              # suivi
sky autostop liquid-train -i 30                             # coupe après 30 min d'inactivité
sky down liquid-train                                       # détruire maintenant
```

Autre config d'entraînement : `--env TRAIN_CONFIG=configs/training/<fichier>.yaml`.
Autre GPU : éditer `accelerators:` (`sky show-gpus --cloud runpod`).

## Creating a serverless endpoint — verified recipe

Settled on 2026-08-22 after four failed deployments. Every value below cost one.

| Field | Value | Why |
|---|---|---|
| Branch | the working branch | not `main` — the Dockerfiles live on the feature branch |
| Dockerfile path | `infra/Dockerfile.serve.liquid` or `infra/Dockerfile.serve` | the console scans the repo root and finds nothing |
| Container disk | **30 GB** (liquid) / **45 GB** (vLLM) | default is 5 GB; the images are 8-12 GB |
| Max workers | 1 | one listener does not need three model loads |
| FlashBoot | on | keeps the worker warm between turns |
| Idle timeout | 5-10 s | longer bills an idle worker after the last turn |

### The four traps, in the order they bite

**1. The endpoint is created before its image exists.** The GitHub build runs
after creation, so the first worker waits on nothing. `INITIALIZING` with no
container logs and `unhealthy: 0` is that, not a slow pull. Check
`list-endpoint-releases`: no `GIT_BUILD` entry means no image.

**2. The container disk is too small.** The image cannot be unpacked, and the
symptom is identical to trap 1 — the container never reaches the point where it
could log anything.

**3. `python:*-slim` has no C compiler.** torch routes some ops to Triton kernels
JIT-compiled on first use. The worker pulls its image, loads the model, then dies
on the first generation. It fails *after* the pull, which reads as slowness.
`build-essential` is the only fix — no environment variable disables this, since
the kernel is reached through `torch._native`, not `torch.compile`.

**4. Floating dependency bounds.** `vllm-omni>=0.22` resolves to the latest,
which demands a newer vLLM than the wheel pinned by URL above it, and the build
fails on the conflict. Pin the range.

### Validate an install without paying for a build

A failed image build costs twenty minutes; reproducing the same install on a
Colab L4 costs three, and yields the resolver's actual error:

```bash
colab new -s deps-check --gpu L4
colab exec -s deps-check --timeout 900 -f scripts/check_deps.py   # dry-run installs
colab stop -s deps-check
```

### Reading a stuck worker

`get-job-status` names the failing worker and says whether the queue is a
capacity problem or a crash loop — `endpoint-health` alone does not.

```bash
# workers, their image tag, and their state
mcp: list-endpoint-workers <endpoint>
# what the container actually printed
mcp: stream-worker-logs <endpoint> <workerId> --source container
```
