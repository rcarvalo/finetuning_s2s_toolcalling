"""Client d'inférence distante (endpoint serverless RunPod).

Zéro dépendance GPU : seul ``httpx`` est requis (extra ``client``), ce qui
permet d'exécuter ce module tel quel sur le Reachy Mini.
"""

from lfm2_audio.remote.client import LiquidAudioClient

__all__ = ["LiquidAudioClient"]
