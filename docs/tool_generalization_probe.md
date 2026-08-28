# Sonde de généralisation d'outils — le modèle lit-il la liste déclarée ?

Question : v4/v5, entraînés sur `web_search` + `db_query` uniquement,
appellent-ils correctement un outil **jamais vu**, déclaré seulement dans le
system prompt ? Tout le plan agentique (MCP, LangGraph, ADK, un outil
`delegate`) en dépend.

Protocole : 38 énoncés parlés (Kokoro, 8 voix), quatre déclarations d'outils
sur **les mêmes** énoncés — seule la liste change. `benchmark/tool_probe/`,
runner `infra/sky_probe_tools.yaml`.

## Résultats

Score agrégé `tool_call` :

| Condition | v4 | v5 |
|---|---|---|
| contrôle (2 entraînés) | 0,833 | 0,889 |
| +1 inédit (`delegate`) | 0,682 | 0,727 |
| +4 inédits | 0,559 | 0,588 |
| inédits seuls | 0,462 | 0,538 |

Décomposition par type de cas — c'est elle qui porte le verdict :

| | condition | **inédits** | entraînés | négatifs | hallucination |
|---|---|---|---|---|---|
| v4 | contrôle | — | 12/12 | 6/6 | 0/18 |
| v4 | +1 (`delegate`) | **0/4** | 12/12 | 6/6 | 0/22 |
| v4 | +4 | 5/16 (0,31) | 12/12 | 6/6 | 0/34 |
| v4 | inédits seuls | 7/20 (0,35) | — | 6/6 | 0/26 |
| v5 | contrôle | — | 12/12 | 6/6 | 0/18 |
| v5 | +1 (`delegate`) | **0/4** | 12/12 | 6/6 | 0/22 |
| v5 | +4 | 7/16 (0,44) | 12/12 | 6/6 | 0/34 |
| v5 | inédits seuls | 9/20 (0,45) | — | 6/6 | 0/26 |

Validité : le contrôle donne 0,833 sur v4, le score de référence. La sonde
mesure le modèle, pas un artefact de construction.

## Trois faits, dont deux rassurants

**1. Zéro hallucination, partout.** Le modèle n'appelle JAMAIS un outil qu'on
ne lui a pas déclaré — même en « inédits seuls », où `web_search` et
`db_query` sont absents, il ne les invente pas. Il lit donc bien la liste.

**2. Ajouter des outils ne casse rien.** 12/12 sur les cas entraînés et 6/6
sur les négatifs dans TOUTES les conditions. Déclarer quatre outils de plus ne
dégrade pas d'un iota ce qui marche.

**3. Mais il ne comprend pas la sémantique des outils inédits** — 0,31 à 0,45,
et les erreurs sont parlantes : en « inédits seuls », `set_timer` est appelé
pour « planifie-moi un voyage à Lisbonne », `calendar_lookup` pour « analyse
les ventes du trimestre ». Il choisit un outil déclaré au hasard plausible
plutôt que d'après sa description.

## Le point bloquant : `delegate` obtient 0/4, et on sait pourquoi

Dès que les outils entraînés sont disponibles, le modèle les préfère
systématiquement :

    « Planifie un voyage de trois jours à Lisbonne… »
      → web_search(query="Lisbon flight options hotel and restaurant…")
    « Recherche nos trois concurrents et résume leurs prix »
      → db_query(question="What are the three biggest competitors…")

Ces choix ne sont pas absurdes — ils sont **exactement ce que le prompt
d'entraînement ordonne** : « use web_search for anything PUBLIC or CURRENT…
use db_query ONLY for our INTERNAL company data… **Call at most one tool** »
(`core/chat_format.py:62`). Le modèle applique une règle apprise par NOM, qui
ne laisse aucune place à un troisième outil.

C'est fatal pour la topologie retenue (front-end vocal + délégation) :
`delegate` doit l'emporter sur `web_search` sur les tâches multi-étapes.

Note : v5 produit en plus des appels **malformés** en « inédits seuls »
(`[set_timer(duration_minutes=60), "start_timer"), "send_message",…`),
cohérent avec son instabilité déjà constatée.

## Verdict

**Porte non franchie.** Cible : routage inédit ≥ 0,8 en +1 ; mesuré **0,00**.
La couche agentique ne roule pas telle quelle sur les poids actuels.

## Ce qu'il faut essayer AVANT un v6 (≈ 0,5 €, une heure)

Le prompt système est une variable non testée, et c'est lui qui dicte la
préférence pour `web_search`. Deux conditions supplémentaires, sans aucun
réentraînement :

1. **Instructions génériques** : remplacer les règles par nom par « choisis
   l'outil dont la description correspond ; n'appelle rien si aucun ne
   convient », outils inchangés.
2. **Instructions génériques + précédence explicite de `delegate`** pour les
   demandes multi-étapes.

Si le routage inédit remonte, la couche agentique est accessible par prompt et
orchestrateur seuls. Sinon, **v6 « diversité d'outils »** : corpus généré avec
un pool d'outils synthétiques variés (`cli/data/generate.py` accepte déjà des
`tool_definitions` arbitraires) pour apprendre à LIRE un schéma au lieu de
mémoriser deux noms.
