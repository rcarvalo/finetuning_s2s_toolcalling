"""Exceptions du domaine. Toute erreur levée par le paquet en hérite.

Pas de ``raise Exception(...)`` générique dans le code : l'appelant doit pouvoir
distinguer « checkpoint introuvable » de « backend non installé » sans parser un
message.
"""

from __future__ import annotations


class Lfm2AudioError(Exception):
    """Racine des erreurs du paquet."""


class CheckpointError(Lfm2AudioError):
    """Checkpoint introuvable, incomplet, ou de layout non reconnu."""


class BackendUnavailableError(Lfm2AudioError):
    """Backend d'inférence non installé (vLLM-Omni ou liquid-audio absent)."""


class PromptError(Lfm2AudioError):
    """Prompt impossible à rendre (contrat multimodal violé)."""


class ConversionError(CheckpointError):
    """Conversion d'un checkpoint vers le layout vLLM-Omni impossible."""


class RemoteInferenceError(Lfm2AudioError):
    """Échec d'un appel à l'endpoint d'inférence distant (RunPod serverless)."""


class ExportError(Lfm2AudioError):
    """Export de checkpoint impossible : clés inattendues, adaptateur non fusionné."""
