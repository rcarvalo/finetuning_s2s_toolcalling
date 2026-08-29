
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

## 29/08 — la moitié du budget dans les chemins d'échec jamais exercés
- **Aucun job payant sans avoir exercé son chemin d'échec** : tuer le process en
  plein run dans un test (le flush par famille a perdu 2000 dialogues payés 2×),
  simuler la fin de job (les pods sans auto-stop ont brûlé ~5 $ à vide).
- **Un correctif ne s'applique qu'à un patient malade** : sonder avant de
  réparer (le réalignement torchaudio a cassé le Colab sain qu'il devait imiter).
- **Avant d'accuser l'environnement, chercher ce que NOTRE code injecte** :
  six runs Voxtral perdus sur trois faux diagnostics, le coupable était notre
  entry point vllm_omni chargé dans tous les process.
