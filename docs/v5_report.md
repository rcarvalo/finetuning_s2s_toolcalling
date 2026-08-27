# v5 — payloads réalistes et refus situés : **ne passe pas la gate**

Adaptateur : `Rcarvalo/lfm25-tc-en-v5-adapter` (1970 steps, mêmes
hyperparamètres que v4 — seules les données changent).

## Chiffres

| | v3 | v4 | **v5** |
|---|---|---|---|
| fresh (tool_call) | 0,823 | 0,830 | **0,830** |
| pertinence | 4,33 | 3,77 | **3,62** |
| ancrage | 2,83 | 4,23 | **3,23** |
| honnêteté | — | 3,15 | **2,85** |
| cohérence | 5,00 | 4,85 | **4,23** |

Non-régression du choix d'outil confirmée. **Régression sur les quatre
critères de réponse**, l'ancrage perdant un point entier — c'est-à-dire ce que
v4 avait le mieux corrigé.

## Ce qui a marché

Là où v5 répond, elle répond mieux que v4 :

- **capitale de l'Italie** → « The capital of Italy is Rome. » (v4 appelait un
  outil *et* refusait) ;
- **Ballon d'Or** → « Johan Cruyff, Michel Platini, and Marco van Basten have
  each won the award three times. » (v4 refusait) ;
- météo + suite (« faut-il un parapluie ? ») → répondu, cohérent, 5/5 partout.

Les payloads en prose ont donc bien appris au modèle à extraire une réponse
d'un texte bruité. L'hypothèse de fond était juste.

## Ce qui a cassé — et c'est un défaut de conception, pas un réglage

Les refus produits sont **auto-contradictoires** :

    I found current price of gold in there, but nothing that mentions
    current price of gold.

    I found latest news about humanoid robots in there, but nothing that
    mentions latest news about humanoid robots.

**Cause mesurée.** Sur les 213 refus d'entraînement nommant deux sujets,
**210 (99 %) nomment des sujets partageant des mots**, parfois presque tous :

    « current status of order number o-45678 »
      vs « current delivery status of order number 78901 »   → 5 mots communs
    « phone number for john smith in the sales »
      vs « phone number for customer innovate solutions »    → 3 mots communs

Les deux changements de v5 **s'annulent** : les distracteurs sont topiquement
proches *par construction*, donc le sujet « trouvé » ressemble nécessairement
au sujet « demandé ». Nommer les deux côtés produit alors des exemples où les
deux côtés sont la même chose, et le modèle les a fusionnés.

Aggravant : la garde `_found_topic` interdisait à l'origine tout recouvrement.
Je l'ai **relâchée** (n'exclure que sous-ensemble/sur-ensemble) parce qu'elle
faisait que la forme riche ne se déclenchait presque jamais. La rareté était le
comportement CORRECT : avec des distracteurs proches, il n'y a le plus souvent
aucun second sujet réellement distinct à nommer.

## v5.1 — une seule variable

1. **Rétablir la garde stricte** : ne nommer le sujet trouvé que s'il ne
   partage AUCUN mot plein avec le sujet demandé. Sinon, `ASKED_ONLY`.
2. Ne rien changer d'autre. Les payloads en prose et les distracteurs proches
   sont acquis et bénéfiques — la preuve est dans les réponses ci-dessus.

Vérification avant entraînement : compter les refus à deux sujets et leur taux
de recouvrement. Attendu ≈ 0 % de recouvrement, et une forte majorité de refus
en forme `ASKED_ONLY`.
