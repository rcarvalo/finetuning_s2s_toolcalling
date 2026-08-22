# Devcontainer

Image **CPU** par défaut : lint, typecheck, tests non-GPU, préparation de
données, orchestrateur (backends mockés). C'est tout ce qui tourne sur un Mac.

```bash
make check      # lint + typecheck + tests — doit passer dans le container
```

## Ce qui exige un GPU

L'extra `serving` (vLLM-Omni + liquid-audio) ne s'installe pas sur une image
CPU : le wheel vLLM est CUDA-only. Deux options :

1. **Hôte NVIDIA** (pod, workstation Linux) : décommenter `runArgs` dans
   `devcontainer.json`, puis `make install-serving`.
2. **Colab / pod distant** : voir `docs/serving.md`. Les notebooks de
   `notebooks/` sont faits pour ça.

## Note vLLM 0.22 + CUDA 12

Le wheel PyPI de vLLM 0.22 est buildé en CUDA 13. Sur un hôte CUDA 12 (Colab),
installer le wheel `+cu129` du release — `lfm2_audio.core.env.require_vllm()`
affiche la commande exacte quand l'import échoue.
