
## 2026-08-24 — Rollout d'un contrat de fil strict
Le protocole client↔worker refuse les champs inconnus (voulu). Conséquence :
tout nouveau champ se déploie SERVEUR D'ABORD (image basculée), CLIENT ENSUITE.
Livrer le client en premier casse l'app pendant toute la fenêtre de build —
c'est arrivé avec `history` (multi-tours) : tours 2+ en réponse vide ~30 min.

## 2026-08-24 — Historique à sens unique
Rejouer seulement les réponses de l'assistant (tours user vides, faute de
transcription) fait mélanger au modèle des sujets sans rapport : il voit ses
propres réponses sans savoir ce qu'on lui a demandé. Cap à un échange en
attendant que le worker renvoie une transcription du tour utilisateur.
