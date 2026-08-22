"""Implémentations concrètes des backends.

Jamais importées directement : passer par ``LFM2Audio.from_pretrained`` ou par
le registre — les imports lourds (vLLM, liquid-audio) sont ainsi différés.
"""
