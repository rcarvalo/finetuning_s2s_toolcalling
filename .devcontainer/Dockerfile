# Image de dev CPU : tout ce qui tourne sans GPU (lint, typecheck, tests,
# préparation de données, orchestrateur mocké). Le serving vLLM-Omni exige un
# GPU NVIDIA — cf. .devcontainer/README.md pour le variant CUDA.
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# libsndfile1 + ffmpeg : requis par soundfile / torchaudio (lecture WAV, resample).
# postgresql-client : inspection de la base de l'outil query_database.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        postgresql-client \
        direnv \
        make \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

# uv : seul gestionnaire de paquets autorisé sur ce projet.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# L'env du projet vit dans le workspace (monté), pas dans l'image.
ENV UV_PROJECT_ENVIRONMENT=/workspaces/lfm2-audio/.venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
